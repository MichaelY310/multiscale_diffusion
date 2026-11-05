#!/usr/bin/env bash
# Paper-parity RDD-FM Unified XL/2 run with a job-scoped SF 07:00 watchdog.
#
# DDPM allocation is written finest -> middle -> coarsest:
#   32x32 / 16x16 / 8x8 = 300 / 300 / 400.
# FM data time runs noise -> data, so the strict mirror is written
# coarsest -> middle -> finest:
#   8x8 / 16x16 / 32x32 = 400 / 300 / 300,
# with boundaries [0.0, 0.4, 0.7, 1.0].
#
# This launcher supports 4 or 8 GPUs. The default server layout is GPUs
# 4,5,6,7 with effective global batch 256. MICRO_BATCH_SIZE is the global
# micro batch per forward pass. On four 96GB GPUs, 256 was measured stable:
# 64/GPU x 4 GPUs x accumulation 1 (about 84.5GB/GPU).
#
# The watchdog stops THIS launch at the next 07:00 in America/Los_Angeles.
# It first creates a unique stop file so all DDP ranks save an exact-step
# checkpoint.  If the job has not exited within SAVE_GRACE_SECONDS, it sends
# TERM and finally KILL only to same-UID processes in the private process group
# created for this launch.  It never uses pkill, killall, a username-wide
# filter, or a command-name-wide filter.
#
# Re-run the same script after a scheduled stop to auto-resume the newest
# corrected-run checkpoint.  The script does not auto-start at another time.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"

readonly GLOBAL_BATCH_SIZE=256
readonly MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-256}"
readonly MAX_TRAIN_STEPS=400000
readonly NUM_WORKERS="${NUM_WORKERS:-4}"
readonly SF_TIMEZONE="America/Los_Angeles"
readonly SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-300}"
readonly TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/checkpoints/RDD-FM-Unified-XL-corrected-300-300-400}"
TRAIN_ENTRY="${REPO_ROOT}/2d/train_RDD_FM_imagenet256_unified.py"
CONTROL_DIR="${RESULTS_DIR}/.sf7-control"
CHECKPOINT_DIR="${RESULTS_DIR}/imagenet256-fm-unified/checkpoints"
LOG_DIR="${REPO_ROOT}/logs"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/train_rdd_fm_unified_xl_corrected_300_300_400_8gpu.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${LOG_DIR}/train_rdd_fm_unified_xl_corrected_300_300_400_sf7_watchdog.log}"

mkdir -p "${RESULTS_DIR}" "${CONTROL_DIR}" "${LOG_DIR}"

# Prevent two copies of this launcher from sharing one result directory.
command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required (normally provided by util-linux)." >&2
    exit 1
}
exec 9>"${CONTROL_DIR}/launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another corrected FM launcher already owns ${CONTROL_DIR}/launcher.lock" >&2
    exit 1
fi

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
    echo "ERROR: training entry not found: ${TRAIN_ENTRY}" >&2
    exit 1
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != 4 && ${#GPU_IDS[@]} != 8 )); then
    echo "ERROR: this parity launcher requires exactly 4 or 8 GPU ids; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % ${#GPU_IDS[@]} != 0 )); then
    echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} is not divisible by ${#GPU_IDS[@]} GPUs" >&2
    exit 1
fi
if (( MICRO_BATCH_SIZE <= 0 || GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
    echo "ERROR: global batch ${GLOBAL_BATCH_SIZE} must be divisible by global micro batch ${MICRO_BATCH_SIZE}" >&2
    exit 1
fi
if (( MICRO_BATCH_SIZE % ${#GPU_IDS[@]} != 0 )); then
    echo "ERROR: global micro batch ${MICRO_BATCH_SIZE} is not divisible by ${#GPU_IDS[@]} GPUs" >&2
    exit 1
fi

# Activate the dedicated training environment first.  The eval environment is
# deliberately not selected here: it may contain PyTorch for sampling but no
# torchrun/training dependencies.
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

for required_command in python3 torchrun setsid ps awk tee; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        echo "ERROR: required command is unavailable: ${required_command}" >&2
        exit 1
    }
done

# Fail before allocating GPUs if the IANA timezone database is unavailable.
python3 -c 'from zoneinfo import ZoneInfo; ZoneInfo("America/Los_Angeles")' || {
    echo "ERROR: Python cannot load the America/Los_Angeles timezone." >&2
    exit 1
}

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/imagenet}"
if [[ -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train" ]]; then
    DATA_PATH="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train"
elif [[ -d "${DATA_ROOT}/train" ]]; then
    DATA_PATH="${DATA_ROOT}/train"
else
    echo "ERROR: original ImageNet train directory was not found below ${DATA_ROOT}" >&2
    exit 1
fi

VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
if [[ ! -d "${VAE_PATH}" ]]; then
    echo "ERROR: local SD VAE-EMA directory not found: ${VAE_PATH}" >&2
    echo "Run the repository setup/download script before training." >&2
    exit 1
fi

# Select an unused local rendezvous port unless the caller pins one.
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT="$({ python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1]); s.close()'; })"
fi

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
    if [[ ! -f "${RESUME_CKPT}" ]]; then
        echo "ERROR: RESUME_CKPT does not exist: ${RESUME_CKPT}" >&2
        exit 1
    fi
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
elif [[ "${AUTO_RESUME:-1}" == "1" && -d "${CHECKPOINT_DIR}" ]]; then
    LATEST_CKPT="$({ find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '[0-9]*.pt' -print | sort -V | tail -n 1; })"
    if [[ -n "${LATEST_CKPT}" ]]; then
        RESUME_ARGS=(--resume "${LATEST_CKPT}")
    fi
fi

RUN_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STOP_FILE="${CONTROL_DIR}/stop-${RUN_TOKEN}"
PID_RECORD="${CONTROL_DIR}/process-${RUN_TOKEN}.txt"

TRAIN_ARGS=(
    --model DiT-XL/2
    --resolutions 32,16,8
    --stage-lengths 400,300,300
    --diffusion-steps 1000
    --results-dir "${RESULTS_DIR}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --num-workers "${NUM_WORKERS}"
    --no-pin-memory
    --epochs 1400
    --log-every 100
    --ckpt-every 5000
    --vae ema
    --vae-path "${VAE_PATH}"
    --global-seed 0
    --time-sampler uniform
    --data-source imagenet
    --data-path "${DATA_PATH}"
    --stop-file "${STOP_FILE}"
    "${RESUME_ARGS[@]}"
)

TRAIN_CMD=(
    torchrun
    --nnodes=1
    --nproc_per_node="${#GPU_IDS[@]}"
    --master_port="${MASTER_PORT}"
    "${TRAIN_ENTRY}"
    "${TRAIN_ARGS[@]}"
)

watchdog_log() {
    local message="$1"
    printf '[%s] %s\n' "$(date -Is)" "${message}" | tee -a "${WATCHDOG_LOG}"
}

owned_group_pids() {
    local process_group="$1"
    ps -eo pid=,pgid=,uid= | awk \
        -v wanted_pgid="${process_group}" -v wanted_uid="${UID}" \
        '$2 == wanted_pgid && $3 == wanted_uid { print $1 }'
}

group_is_alive() {
    local process_group="$1"
    [[ -n "$(owned_group_pids "${process_group}")" ]]
}

signal_owned_group() {
    local signal_name="$1"
    local process_group="$2"
    local process_ids=()
    mapfile -t process_ids < <(owned_group_pids "${process_group}")
    for process_id in "${process_ids[@]}"; do
        # The UID and private PGID were both verified above.  Signal PIDs
        # individually so a foreign-UID process could never be targeted even
        # in the pathological case of a mixed-ownership process group.
        kill "-${signal_name}" "${process_id}" 2>/dev/null || true
    done
}

seconds_until_next_sf_7am() {
    python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=7, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
seconds = max(0, int((target - now).total_seconds()))
print(seconds, target.isoformat())
PY
}

sf7_watchdog() {
    local process_group="$1"
    local delay_seconds target_time
    read -r delay_seconds target_time < <(seconds_until_next_sf_7am)
    watchdog_log "Watching private PGID ${process_group}; next SF stop is ${target_time} (${delay_seconds}s)."
    sleep "${delay_seconds}"

    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training group ${process_group} exited before the scheduled stop."
        return 0
    fi

    printf 'requested_at=%s\ntimezone=%s\npgid=%s\n' \
        "$(date -Is)" "${SF_TIMEZONE}" "${process_group}" > "${STOP_FILE}"
    watchdog_log "SF 07:00 reached; requested clean checkpoint-and-stop via ${STOP_FILE}."

    local deadline=$((SECONDS + SAVE_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do
        sleep 1
    done
    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training stopped cleanly after writing its exact-step checkpoint."
        return 0
    fi

    watchdog_log "Clean-stop grace expired; sending TERM only to UID ${UID}, PGID ${process_group}."
    signal_owned_group TERM "${process_group}"
    deadline=$((SECONDS + TERM_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do
        sleep 1
    done
    if group_is_alive "${process_group}"; then
        watchdog_log "TERM grace expired; sending KILL only to UID ${UID}, PGID ${process_group}."
        # Repeat enumeration once so every same-UID member still present in the
        # private group is covered even if torchrun was shutting workers down.
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
        # A manual launcher interruption still remains job-scoped.
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

echo "============================================================"
echo "RDD-FM Unified XL/2 — corrected DDPM 300/300/400 mirror"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}; global batch: ${GLOBAL_BATCH_SIZE}"
echo "Per-GPU micro batch: $((MICRO_BATCH_SIZE / ${#GPU_IDS[@]})); accumulation: $((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))"
echo "FM coarse/middle/fine: 400/300/300; boundaries: [0.0, 0.4, 0.7, 1.0]"
echo "Original ImageNet: ${DATA_PATH}"
echo "Local VAE-EMA: ${VAE_PATH}"
echo "Target optimizer step: ${MAX_TRAIN_STEPS}"
echo "Training log: ${TRAIN_LOG}"
echo "Watchdog log: ${WATCHDOG_LOG}"
echo "Scheduled stop: next 07:00 America/Los_Angeles"
if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "Resume checkpoint: ${RESUME_ARGS[1]}"
else
    echo "Resume checkpoint: none (training from scratch)"
fi
echo "============================================================"

cd "${REPO_ROOT}"
export TRAIN_LOG
# The inner bash and every torchrun worker inherit a new, private session and
# process group.  tee is inside that same group, so no logging helper is left
# behind after a scheduled hard stop.
setsid bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "${TRAIN_LOG}"' \
    rdd-fm-train "${TRAIN_CMD[@]}" &
TRAIN_PID=$!

sleep 1
TRAIN_PGID="$(ps -o pgid= -p "${TRAIN_PID}" | awk '{print $1}')"
if [[ -z "${TRAIN_PGID}" || "${TRAIN_PGID}" != "${TRAIN_PID}" ]]; then
    echo "ERROR: failed to create a private training process group (pid=${TRAIN_PID}, pgid=${TRAIN_PGID:-missing})." >&2
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
    exit 1
fi
printf 'pid=%s\npgid=%s\nuid=%s\nstop_file=%s\n' \
    "${TRAIN_PID}" "${TRAIN_PGID}" "${UID}" "${STOP_FILE}" > "${PID_RECORD}"

sf7_watchdog "${TRAIN_PGID}" &
WATCHDOG_PID=$!

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""

if [[ -f "${STOP_FILE}" ]]; then
    watchdog_log "Launcher observed the scheduled stop; torchrun exit status was ${TRAIN_STATUS}."
    exit 0
fi
if (( TRAIN_STATUS != 0 )); then
    echo "ERROR: torchrun exited with status ${TRAIN_STATUS}; inspect ${TRAIN_LOG}" >&2
    exit "${TRAIN_STATUS}"
fi

echo "Training completed normally."
