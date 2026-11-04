#!/usr/bin/env bash
# Multistage RDD allocation ablation:
#   32x32 / 16x16 / 8x8 = 600 / 300 / 100
# Everything else matches the native-ImageNet 100/300/600 run:
#   DiT-XL/2 + DiT-L/2 + DiT-B/2, 8 GPUs, global batch 256,
#   uniform local-timestep sampling, no loss weighting, VAE-EMA, seed 0.
#
# A job-scoped watchdog requests exact-step checkpointing at the next 07:00
# in America/Los_Angeles.  After SAVE_GRACE_SECONDS it terminates, and then
# kills, only same-UID processes in this launch's private process group.  It
# never uses pkill, killall, username-wide filters, or command-name filters.
# Re-running this launcher auto-resumes the newest common 32/16/8 checkpoint.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT

readonly GLOBAL_BATCH_SIZE=256
readonly PER_RANK_MICRO_BATCH="${PER_RANK_MICRO_BATCH:-32}"
readonly MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
readonly NUM_WORKERS="${NUM_WORKERS:-4}"
readonly LA_TIMEZONE="America/Los_Angeles"
readonly SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-300}"
readonly TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/checkpoints/RDD-Multistage-600-300-100}"
EXPERIMENT_DIR="${RESULTS_DIR}/imagenet256-multiscale"
TRAIN_ENTRY="${REPO_ROOT}/2d/train_RDD_imagenet256.py"
CONTROL_DIR="${RESULTS_DIR}/.la7-control"
LOG_DIR="${REPO_ROOT}/logs"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train_rdd_multistage_600_300_100_8gpu.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${LOG_DIR}/train_rdd_multistage_600_300_100_la7_watchdog.log}"

mkdir -p "${RESULTS_DIR}" "${CONTROL_DIR}" "${LOG_DIR}"

command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required." >&2
    exit 1
}
exec 9>"${CONTROL_DIR}/launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher owns ${CONTROL_DIR}/launcher.lock" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
    echo "ERROR: training entry not found: ${TRAIN_ENTRY}" >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != 8 )); then
    echo "ERROR: this run requires exactly 8 GPU ids; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi
MICRO_GLOBAL_BATCH=$((PER_RANK_MICRO_BATCH * ${#GPU_IDS[@]}))
if (( PER_RANK_MICRO_BATCH <= 0 || GLOBAL_BATCH_SIZE % MICRO_GLOBAL_BATCH != 0 )); then
    echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} must be divisible by per-rank micro ${PER_RANK_MICRO_BATCH} x 8 GPUs." >&2
    exit 1
fi
readonly GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / MICRO_GLOBAL_BATCH))

if [[ -f "${REPO_ROOT}/.venv-rdd-train/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-train/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

for required_command in python3 torchrun setsid ps awk tee find sort; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        echo "ERROR: required command is unavailable: ${required_command}" >&2
        exit 1
    }
done
python3 -c 'from zoneinfo import ZoneInfo; ZoneInfo("America/Los_Angeles")'

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/imagenet}"
if [[ -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train" ]]; then
    DATA_PATH="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train"
elif [[ -d "${DATA_ROOT}/train" ]]; then
    DATA_PATH="${DATA_ROOT}/train"
else
    echo "ERROR: native ImageNet train directory not found below ${DATA_ROOT}" >&2
    exit 1
fi

VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
if [[ ! -f "${VAE_PATH}/config.json" ]]; then
    echo "ERROR: local SD VAE-EMA not found: ${VAE_PATH}" >&2
    exit 1
fi

if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$({ python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; })"
fi

# Resume only a filename that exists for all three independently trained stages.
RESUME_ARGS=()
LATEST_COMMON=""
if [[ "${AUTO_RESUME:-1}" == "1" && -d "${EXPERIMENT_DIR}/checkpoints_32x32" ]]; then
    while IFS= read -r checkpoint_32; do
        checkpoint_name="$(basename "${checkpoint_32}")"
        if [[ -f "${EXPERIMENT_DIR}/checkpoints_16x16/${checkpoint_name}" \
           && -f "${EXPERIMENT_DIR}/checkpoints_8x8/${checkpoint_name}" ]]; then
            LATEST_COMMON="${checkpoint_name}"
            break
        fi
    done < <(find "${EXPERIMENT_DIR}/checkpoints_32x32" -maxdepth 1 -type f \
        -name '[0-9]*.pt' -print | sort -Vr)
fi
if [[ -n "${LATEST_COMMON}" ]]; then
    RESUME_ARGS=(
        --resume-32 "${EXPERIMENT_DIR}/checkpoints_32x32/${LATEST_COMMON}"
        --resume-16 "${EXPERIMENT_DIR}/checkpoints_16x16/${LATEST_COMMON}"
        --resume-8  "${EXPERIMENT_DIR}/checkpoints_8x8/${LATEST_COMMON}"
    )
fi

RUN_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STOP_FILE="${CONTROL_DIR}/stop-${RUN_TOKEN}"
PID_RECORD="${CONTROL_DIR}/process-${RUN_TOKEN}.txt"

TRAIN_ARGS=(
    --model-32 DiT-XL/2
    --model-16 DiT-L/2
    --model-8 DiT-B/2
    --stage-lengths 600,300,100
    --diffusion-steps 1000
    --results-dir "${RESULTS_DIR}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${PER_RANK_MICRO_BATCH}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --epochs 1400
    --num-workers "${NUM_WORKERS}"
    --log-every 100
    --ckpt-every 5000
    --loss-weighting none
    --timestep-sampler uniform
    --vae ema
    --vae-path "${VAE_PATH}"
    --global-seed 0
    --data-source imagenet
    --data-path "${DATA_PATH}"
    --stop-file "${STOP_FILE}"
    "${RESUME_ARGS[@]}"
)

TRAIN_CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node=8
    --master_port="${MASTER_PORT}"
    "${TRAIN_ENTRY}"
    "${TRAIN_ARGS[@]}"
)

watchdog_log() {
    printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "${WATCHDOG_LOG}"
}

owned_group_pids() {
    local wanted_pgid="$1"
    # torchrun workers may create their own process groups.  Include both the
    # private launcher PGID and same-UID processes carrying this launch's exact
    # unique --stop-file token.  The latter cannot select another run/user.
    ps -eo pid=,pgid=,uid=,args= | awk \
        -v wanted_pgid="${wanted_pgid}" -v wanted_uid="${UID}" \
        -v token="${STOP_FILE}" -v self="$$" \
        '$3 == wanted_uid && $1 != self && ($2 == wanted_pgid || index($0, token)) { print $1 }'
}

group_is_alive() {
    [[ -n "$(owned_group_pids "$1")" ]]
}

signal_owned_group() {
    local signal_name="$1"
    local process_group="$2"
    local process_ids=()
    mapfile -t process_ids < <(owned_group_pids "${process_group}")
    for process_id in "${process_ids[@]}"; do
        kill "-${signal_name}" "${process_id}" 2>/dev/null || true
    done
}

seconds_until_next_la_7am() {
    python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=7, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
}

la7_watchdog() {
    local process_group="$1"
    local delay_seconds target_time
    read -r delay_seconds target_time < <(seconds_until_next_la_7am)
    watchdog_log "Watching private PGID ${process_group}; next LA stop is ${target_time} (${delay_seconds}s)."
    sleep "${delay_seconds}"

    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training group ${process_group} exited before the scheduled stop."
        return 0
    fi

    printf 'requested_at=%s\ntimezone=%s\npgid=%s\n' \
        "$(date -Is)" "${LA_TIMEZONE}" "${process_group}" > "${STOP_FILE}"
    watchdog_log "LA 07:00 reached; requested exact-step checkpoint-and-stop via ${STOP_FILE}."

    local deadline=$((SECONDS + SAVE_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do
        sleep 1
    done
    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training saved all three stages and stopped cleanly."
        return 0
    fi

    watchdog_log "Save grace expired; sending TERM only to UID ${UID}, PGID ${process_group}."
    signal_owned_group TERM "${process_group}"
    deadline=$((SECONDS + TERM_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do
        sleep 1
    done
    if group_is_alive "${process_group}"; then
        watchdog_log "TERM grace expired; sending KILL only to UID ${UID}, PGID ${process_group}."
        signal_owned_group KILL "${process_group}"
        sleep 1
        signal_owned_group KILL "${process_group}"
    fi
}

WATCHDOG_PID=""
TRAIN_PID=""
cleanup_launcher() {
    local exit_status=$?
    trap - EXIT INT TERM
    if [[ -n "${WATCHDOG_PID}" ]] && kill -0 "${WATCHDOG_PID}" 2>/dev/null; then
        kill "${WATCHDOG_PID}" 2>/dev/null || true
        wait "${WATCHDOG_PID}" 2>/dev/null || true
    fi
    if [[ -n "${TRAIN_PID}" ]] && kill -0 "${TRAIN_PID}" 2>/dev/null; then
        local recorded_pgid
        recorded_pgid="$(awk -F= '$1 == "pgid" {print $2}' "${PID_RECORD}" 2>/dev/null || true)"
        if [[ -n "${recorded_pgid}" ]]; then
            signal_owned_group TERM "${recorded_pgid}"
        fi
    fi
    rm -f -- "${PID_RECORD}"
    exit "${exit_status}"
}
trap cleanup_launcher EXIT INT TERM

read -r _delay NEXT_LA_STOP < <(seconds_until_next_la_7am)
echo "============================================================"
echo "RDD Multistage 600/300/100 (32/16/8)"
echo "Models: DiT-XL/2 + DiT-L/2 + DiT-B/2"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}; global batch: ${GLOBAL_BATCH_SIZE}"
echo "Per-rank micro batch: ${PER_RANK_MICRO_BATCH}; accumulation: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Original ImageNet: ${DATA_PATH}"
echo "Target optimizer step: ${MAX_TRAIN_STEPS}"
echo "Results: ${RESULTS_DIR}"
echo "Training log: ${TRAIN_LOG}"
echo "Watchdog log: ${WATCHDOG_LOG}"
echo "Scheduled stop: ${NEXT_LA_STOP}"
if [[ -n "${LATEST_COMMON}" ]]; then
    echo "Auto-resume common checkpoint: ${LATEST_COMMON}"
else
    echo "Resume checkpoint: none (training from scratch)"
fi
echo "============================================================"

cd "${REPO_ROOT}"
export TRAIN_LOG
setsid bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "${TRAIN_LOG}"' \
    rdd-ms600300100-train "${TRAIN_CMD[@]}" &
TRAIN_PID=$!

sleep 1
TRAIN_PGID="$(ps -o pgid= -p "${TRAIN_PID}" | awk '{print $1}')"
if [[ -z "${TRAIN_PGID}" || "${TRAIN_PGID}" != "${TRAIN_PID}" ]]; then
    echo "ERROR: failed to create private process group (pid=${TRAIN_PID}, pgid=${TRAIN_PGID:-missing})." >&2
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
    exit 1
fi
printf 'pid=%s\npgid=%s\nuid=%s\nstop_file=%s\n' \
    "${TRAIN_PID}" "${TRAIN_PGID}" "${UID}" "${STOP_FILE}" > "${PID_RECORD}"

la7_watchdog "${TRAIN_PGID}" &
WATCHDOG_PID=$!

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""

if [[ -f "${STOP_FILE}" ]]; then
    watchdog_log "Launcher observed scheduled stop; torchrun exit status was ${TRAIN_STATUS}."
    exit 0
fi
if (( TRAIN_STATUS != 0 )); then
    echo "ERROR: torchrun exited with status ${TRAIN_STATUS}; inspect ${TRAIN_LOG}" >&2
    exit "${TRAIN_STATUS}"
fi

echo "Training completed normally."
