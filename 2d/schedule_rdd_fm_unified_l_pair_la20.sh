#!/usr/bin/env bash
# One-shot server-side timer for the two RDD-FM Unified-L allocation runs.
# At the next 20:00 in America/Los_Angeles, each run resumes from its newest
# numeric checkpoint.  The scheduler never signals an existing process; if a
# requested GPU group is occupied, it waits until that exact group is free.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TIMEZONE="America/Los_Angeles"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRAINER="${REPO_ROOT}/2d/train_RDD_FM_unified_L_allocation_4gpu.sh"
SCHEDULER_LOG="${REPO_ROOT}/logs/schedule_rdd_fm_unified_l_pair_la20.log"

mkdir -p "${REPO_ROOT}/logs"
[[ -f "${TRAINER}" ]] || { echo "ERROR: missing trainer ${TRAINER}" >&2; exit 1; }

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${SCHEDULER_LOG}"
}

read -r DELAY_SECONDS TARGET_ISO < <(python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=20, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
)

gpu_group_is_free() {
    local requested_csv="$1"
    local requested=",${requested_csv},"
    local index uuid app_uuid app_pid
    declare -A selected_uuids=()

    while IFS=',' read -r index uuid; do
        index="${index//[[:space:]]/}"
        uuid="${uuid//[[:space:]]/}"
        if [[ "${requested}" == *",${index},"* ]]; then
            selected_uuids["${uuid}"]=1
        fi
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)

    if (( ${#selected_uuids[@]} != 4 )); then
        log "ERROR: expected four physical GPU UUIDs for ${requested_csv}."
        return 1
    fi

    while IFS=',' read -r app_uuid app_pid; do
        app_uuid="${app_uuid//[[:space:]]/}"
        app_pid="${app_pid//[[:space:]]/}"
        if [[ -n "${selected_uuids["${app_uuid}"]:-}" ]]; then
            log "GPUs ${requested_csv} occupied by PID ${app_pid}; waiting without signaling it."
            return 1
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
    return 0
}

matching_trainer_is_alive() {
    local result_dir="$1"
    ps -u "$(id -u)" -o args= \
        | grep -F -- "--results-dir ${result_dir}" \
        | grep -v grep >/dev/null
}

launch_when_free() {
    local session="$1"
    local gpu_ids="$2"
    local allocation="$3"
    local result_dir="$4"
    local checkpoint_dir="${result_dir}/imagenet256-fm-unified/checkpoints"
    local latest_checkpoint

    if matching_trainer_is_alive "${result_dir}"; then
        log "${result_dir} is already running under this UID; refusing a duplicate launch."
        return 0
    fi

    latest_checkpoint="$(find "${checkpoint_dir}" -maxdepth 1 -type f -name '[0-9]*.pt' -print | sort -V | tail -n 1)"
    [[ -n "${latest_checkpoint}" && -s "${latest_checkpoint}" ]] || {
        log "ERROR: no stable numeric checkpoint in ${checkpoint_dir}."
        return 1
    }

    while ! gpu_group_is_free "${gpu_ids}"; do
        sleep "${POLL_SECONDS}"
    done

    if matching_trainer_is_alive "${result_dir}"; then
        log "${result_dir} started while GPUs were being polled; refusing a duplicate launch."
        return 0
    fi
    if tmux has-session -t "${session}" 2>/dev/null; then
        log "ERROR: tmux session already exists: ${session}; refusing to replace it."
        return 1
    fi

    tmux new-session -d -s "${session}" \
        "cd '${REPO_ROOT}' && AUTO_RESUME=1 CUDA_VISIBLE_DEVICES='${gpu_ids}' DDPM_ALLOCATION='${allocation}' MICRO_BATCH_SIZE=256 bash '${TRAINER}'"
    log "Launched ${allocation} in ${session} on GPUs ${gpu_ids}; resume=${latest_checkpoint}."
}

log "One-shot timer armed for ${TARGET_ISO} (${DELAY_SECONDS}s from now)."
sleep "${DELAY_SECONDS}"
log "Los Angeles 20:00 reached; checking the two GPU groups."

launch_when_free \
    rdd_fm_unified_l_300300400 \
    0,1,2,3 \
    300,300,400 \
    "${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-300-300-400" &
pid_a=$!

launch_when_free \
    rdd_fm_unified_l_400300300 \
    4,5,6,7 \
    400,300,300 \
    "${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-400-300-300" &
pid_b=$!

status=0
wait "${pid_a}" || status=1
wait "${pid_b}" || status=1
if (( status == 0 )); then
    log "Both scheduled launch requests completed successfully."
else
    log "ERROR: at least one scheduled launch failed; inspect this log."
fi
exit "${status}"
