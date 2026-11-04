#!/usr/bin/env bash
# Shared paper-parity launcher for four-GPU Unified DiT boundary ablations.
#
# The wrapper supplies the finest/middle/coarsest allocation, GPU ids, result
# name, and rendezvous port.  This launcher always uses original ImageNet,
# global batch 256, uniform timestep sampling, legacy unweighted RDD loss, and
# one DiTMultiRes model for every configured resolution.
#
# By default, at the next 07:00 in America/Los_Angeles, a job-scoped watchdog
# asks the trainer to save an exact-step checkpoint and exit.  A caller may set
# STOP_AT_EPOCH to one explicit Unix timestamp (for example, tomorrow at 09:00
# LA time).  If a clean stop stalls, the watchdog signals only same-UID PIDs in
# the private process group made for this run.

set -Eeuo pipefail

: "${RDD_UNIFIED_STAGE_LENGTHS:?wrapper must set RDD_UNIFIED_STAGE_LENGTHS}"
: "${RDD_UNIFIED_RUN_NAME:?wrapper must set RDD_UNIFIED_RUN_NAME}"
: "${RDD_UNIFIED_DEFAULT_GPUS:?wrapper must set RDD_UNIFIED_DEFAULT_GPUS}"
: "${RDD_UNIFIED_DEFAULT_MASTER_PORT:?wrapper must set RDD_UNIFIED_DEFAULT_MASTER_PORT}"
: "${RDD_UNIFIED_LOG_STEM:?wrapper must set RDD_UNIFIED_LOG_STEM}"

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-${RDD_UNIFIED_DEFAULT_GPUS}}"

readonly STAGE_LENGTHS="${RDD_UNIFIED_STAGE_LENGTHS}"
readonly RESOLUTIONS="${RDD_UNIFIED_RESOLUTIONS:-32,16,8}"
readonly MODEL="${RDD_UNIFIED_MODEL:-DiT-L/2}"
readonly RUN_NAME="${RDD_UNIFIED_RUN_NAME}"
readonly GLOBAL_BATCH_SIZE=256
readonly MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-256}"
readonly MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
readonly CKPT_EVERY="${CKPT_EVERY:-5000}"
readonly NUM_WORKERS="${NUM_WORKERS:-4}"
readonly SF_TIMEZONE="America/Los_Angeles"
readonly SAVE_GRACE_SECONDS="${SAVE_GRACE_SECONDS:-300}"
readonly TERM_GRACE_SECONDS="${TERM_GRACE_SECONDS:-20}"
readonly STOP_AT_EPOCH="${STOP_AT_EPOCH:-}"
readonly DISABLE_SCHEDULED_STOP="${DISABLE_SCHEDULED_STOP:-0}"

if [[ "${DISABLE_SCHEDULED_STOP}" != "0" && "${DISABLE_SCHEDULED_STOP}" != "1" ]]; then
    echo "ERROR: DISABLE_SCHEDULED_STOP must be 0 or 1" >&2
    exit 1
fi

TRAIN_ENTRY="${REPO_ROOT}/2d/train_RDD_imagenet256_unified.py"
RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/checkpoints/${RUN_NAME}}"
CHECKPOINT_DIR="${RESULTS_DIR}/imagenet256-unified/checkpoints"
LOG_DIR="${REPO_ROOT}/logs"
TRAIN_LOG="${TRAIN_LOG:-${LOG_DIR}/${RDD_UNIFIED_LOG_STEM}.log}"
WATCHDOG_LOG="${WATCHDOG_LOG:-${LOG_DIR}/${RDD_UNIFIED_LOG_STEM}_sf7_watchdog.log}"
CONTROL_DIR="${RESULTS_DIR}/.control"

mkdir -p "${RESULTS_DIR}" "${LOG_DIR}" "${CONTROL_DIR}"
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required" >&2; exit 1; }
exec 9>"${CONTROL_DIR}/launcher.lock"
if ! flock -n 9; then
    echo "ERROR: another launcher already owns ${CONTROL_DIR}/launcher.lock" >&2
    exit 1
fi

if [[ -f "${REPO_ROOT}/.venv-rdd-train/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-train/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
else
    echo "ERROR: RDD training environment was not found below ${REPO_ROOT}" >&2
    exit 1
fi

for required_command in python3 torchrun setsid ps awk tee; do
    command -v "${required_command}" >/dev/null 2>&1 || {
        echo "ERROR: required command is unavailable: ${required_command}" >&2
        exit 1
    }
done
python3 -c 'from zoneinfo import ZoneInfo; ZoneInfo("America/Los_Angeles")' || {
    echo "ERROR: Python cannot load the America/Los_Angeles timezone" >&2
    exit 1
}

export HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"
export TMPDIR="${TMPDIR:-${REPO_ROOT}/.tmp}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
mkdir -p "${HF_HOME}" "${HF_DATASETS_CACHE}" "${TMPDIR}"

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
if (( ${#GPU_IDS[@]} != 4 )); then
    echo "ERROR: exactly four GPUs are required; got ${CUDA_VISIBLE_DEVICES}" >&2
    exit 1
fi
if (( GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 || MICRO_BATCH_SIZE % ${#GPU_IDS[@]} != 0 )); then
    echo "ERROR: MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} must divide 256 and be divisible by four" >&2
    exit 1
fi

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/imagenet}"
if [[ -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train" ]]; then
    DATA_PATH="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train"
elif [[ -d "${DATA_ROOT}/train" ]]; then
    DATA_PATH="${DATA_ROOT}/train"
else
    echo "ERROR: original ImageNet train directory not found below ${DATA_ROOT}" >&2
    exit 1
fi

VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
[[ -d "${VAE_PATH}" ]] || { echo "ERROR: local VAE not found: ${VAE_PATH}" >&2; exit 1; }
[[ -f "${TRAIN_ENTRY}" ]] || { echo "ERROR: training entry not found: ${TRAIN_ENTRY}" >&2; exit 1; }

RESUME_ARGS=()
if [[ -n "${RESUME_CKPT:-}" ]]; then
    [[ -f "${RESUME_CKPT}" ]] || { echo "ERROR: RESUME_CKPT not found: ${RESUME_CKPT}" >&2; exit 1; }
    RESUME_ARGS=(--resume "${RESUME_CKPT}")
elif [[ "${AUTO_RESUME:-1}" == "1" && -d "${CHECKPOINT_DIR}" ]]; then
    LATEST_CKPT="$(find "${CHECKPOINT_DIR}" -maxdepth 1 -type f -name '[0-9]*.pt' -print | sort -V | tail -n 1)"
    [[ -z "${LATEST_CKPT}" ]] || RESUME_ARGS=(--resume "${LATEST_CKPT}")
fi

MASTER_PORT="${MASTER_PORT:-${RDD_UNIFIED_DEFAULT_MASTER_PORT}}"
RUN_TOKEN="$(date -u +%Y%m%dT%H%M%SZ)-$$"
STOP_FILE="${CONTROL_DIR}/stop-${RUN_TOKEN}"
PID_RECORD="${CONTROL_DIR}/process-${RUN_TOKEN}.txt"

TRAIN_ARGS=(
    --model "${MODEL}"
    --resolutions "${RESOLUTIONS}"
    --stage-lengths "${STAGE_LENGTHS}"
    --results-dir "${RESULTS_DIR}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --num-workers "${NUM_WORKERS}"
    --no-pin-memory
    --diffusion-steps 1000
    --epochs 1400
    --log-every 100
    --ckpt-every "${CKPT_EVERY}"
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
    --nproc_per_node="${#GPU_IDS[@]}"
    --master_port="${MASTER_PORT}"
    "${TRAIN_ENTRY}"
    "${TRAIN_ARGS[@]}"
)

watchdog_log() {
    printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "${WATCHDOG_LOG}"
}

owned_group_pids() {
    local process_group="$1"
    ps -eo pid=,pgid=,uid= | awk \
        -v wanted_pgid="${process_group}" -v wanted_uid="${UID}" \
        '$2 == wanted_pgid && $3 == wanted_uid { print $1 }'
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

resolve_stop_schedule() {
    python3 - "${STOP_AT_EPOCH}" <<'PY'
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
explicit_epoch = sys.argv[1]
if explicit_epoch:
    try:
        target = datetime.fromtimestamp(int(explicit_epoch), zone)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SystemExit(f"invalid STOP_AT_EPOCH={explicit_epoch!r}: {exc}")
else:
    target = now.replace(hour=7, minute=0, second=0, microsecond=0)
    if target <= now:
        target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
}

scheduled_stop_watchdog() {
    local process_group="$1"
    local delay_seconds target_time
    read -r delay_seconds target_time < <(resolve_stop_schedule)
    watchdog_log "Watching private PGID ${process_group}; LA stop is ${target_time} (${delay_seconds}s)."
    sleep "${delay_seconds}"

    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training group ${process_group} exited before the scheduled stop."
        return 0
    fi

    printf 'requested_at=%s\ntimezone=%s\npgid=%s\n' \
        "$(date -Is)" "${SF_TIMEZONE}" "${process_group}" > "${STOP_FILE}"
    watchdog_log "Scheduled LA stop ${target_time} reached; requested exact-step checkpoint and clean stop via ${STOP_FILE}."

    local deadline=$((SECONDS + SAVE_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do sleep 1; done
    if ! group_is_alive "${process_group}"; then
        watchdog_log "Training stopped cleanly after its exact-step checkpoint."
        return 0
    fi

    watchdog_log "Save grace expired; TERM only UID ${UID}, private PGID ${process_group}."
    signal_owned_group TERM "${process_group}"
    deadline=$((SECONDS + TERM_GRACE_SECONDS))
    while group_is_alive "${process_group}" && (( SECONDS < deadline )); do sleep 1; done
    if group_is_alive "${process_group}"; then
        watchdog_log "TERM grace expired; KILL only UID ${UID}, private PGID ${process_group}."
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
        [[ -z "${recorded_pgid}" ]] || signal_owned_group TERM "${recorded_pgid}"
    fi
    rm -f -- "${PID_RECORD}"
    exit "${exit_status}"
}
trap cleanup_launcher EXIT INT TERM

echo "============================================================" | tee -a "${TRAIN_LOG}"
echo "${RUN_NAME}" | tee -a "${TRAIN_LOG}"
echo "GPUs=${CUDA_VISIBLE_DEVICES}; model=${MODEL}; resolutions=${RESOLUTIONS}; stage lengths=${STAGE_LENGTHS}" | tee -a "${TRAIN_LOG}"
echo "global batch=${GLOBAL_BATCH_SIZE}; global micro=${MICRO_BATCH_SIZE}; per-GPU micro=$((MICRO_BATCH_SIZE / 4)); accumulation=$((GLOBAL_BATCH_SIZE / MICRO_BATCH_SIZE))" | tee -a "${TRAIN_LOG}"
echo "checkpoint interval=${CKPT_EVERY}; max step=${MAX_TRAIN_STEPS}" | tee -a "${TRAIN_LOG}"
if [[ -n "${STOP_AT_EPOCH}" ]]; then
    echo "scheduled stop=explicit epoch ${STOP_AT_EPOCH} in America/Los_Angeles" | tee -a "${TRAIN_LOG}"
elif [[ "${DISABLE_SCHEDULED_STOP}" == "1" ]]; then
    echo "scheduled stop=disabled; train until step ${MAX_TRAIN_STEPS}" | tee -a "${TRAIN_LOG}"
else
    echo "scheduled stop=next 07:00 America/Los_Angeles" | tee -a "${TRAIN_LOG}"
fi
echo "watchdog log=${WATCHDOG_LOG}" | tee -a "${TRAIN_LOG}"
if (( ${#RESUME_ARGS[@]} > 0 )); then
    echo "resume=${RESUME_ARGS[1]}" | tee -a "${TRAIN_LOG}"
else
    echo "resume=none (from scratch)" | tee -a "${TRAIN_LOG}"
fi
echo "results=${RESULTS_DIR}" | tee -a "${TRAIN_LOG}"
echo "============================================================" | tee -a "${TRAIN_LOG}"

cd "${REPO_ROOT}"
export TRAIN_LOG
setsid bash -c 'set -o pipefail; "$@" 2>&1 | tee -a "${TRAIN_LOG}"' \
    rdd-unified-l-train "${TRAIN_CMD[@]}" &
TRAIN_PID=$!
sleep 1
TRAIN_PGID="$(ps -o pgid= -p "${TRAIN_PID}" | awk '{print $1}')"
if [[ -z "${TRAIN_PGID}" || "${TRAIN_PGID}" != "${TRAIN_PID}" ]]; then
    echo "ERROR: failed to create a private process group (pid=${TRAIN_PID}, pgid=${TRAIN_PGID:-missing})" >&2
    kill "${TRAIN_PID}" 2>/dev/null || true
    wait "${TRAIN_PID}" 2>/dev/null || true
    exit 1
fi
printf 'pid=%s\npgid=%s\nuid=%s\nstop_file=%s\n' \
    "${TRAIN_PID}" "${TRAIN_PGID}" "${UID}" "${STOP_FILE}" > "${PID_RECORD}"

if [[ "${DISABLE_SCHEDULED_STOP}" != "1" ]]; then
    scheduled_stop_watchdog "${TRAIN_PGID}" &
    WATCHDOG_PID=$!
fi

set +e
wait "${TRAIN_PID}"
TRAIN_STATUS=$?
set -e
TRAIN_PID=""

if [[ -f "${STOP_FILE}" ]]; then
    watchdog_log "Launcher observed the scheduled stop; torchrun exit status=${TRAIN_STATUS}."
    exit 0
fi
if (( TRAIN_STATUS != 0 )); then
    echo "ERROR: torchrun exited with status ${TRAIN_STATUS}; inspect ${TRAIN_LOG}" >&2
    exit "${TRAIN_STATUS}"
fi
echo "Training completed normally."
