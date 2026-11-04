#!/usr/bin/env bash
# One-shot server-side timer: resume the two Unified-L allocation ablations at
# the next 21:00 in America/Los_Angeles.  This script never signals a process.
# If a requested GPU group is occupied at the target time, it waits until that
# group is free and then launches only its corresponding job.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/scheduled_resume_unified_l_la21.log}"
TIMEZONE="America/Los_Angeles"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${REPO_ROOT}/logs"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

seconds_until_next_la_21() {
    python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=21, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
}

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
        log "ERROR: could not resolve exactly four GPU UUIDs for ${requested_csv}."
        return 1
    fi

    while IFS=',' read -r app_uuid app_pid; do
        app_uuid="${app_uuid//[[:space:]]/}"
        app_pid="${app_pid//[[:space:]]/}"
        if [[ -n "${app_uuid}" && -n "${selected_uuids["${app_uuid}"]:-}" ]]; then
            log "GPU group ${requested_csv} is occupied (PID ${app_pid}); waiting without signaling it."
            return 1
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
    return 0
}

launch_when_free() {
    local session_name="$1"
    local gpu_ids="$2"
    local launcher="$3"

    while ! gpu_group_is_free "${gpu_ids}"; do
        sleep "${POLL_SECONDS}"
    done

    if tmux has-session -t "${session_name}" 2>/dev/null; then
        log "Session ${session_name} already exists; refusing a duplicate launch."
        return 0
    fi

    tmux new-session -d -s "${session_name}" \
        "cd '${REPO_ROOT}' && AUTO_RESUME=1 CUDA_VISIBLE_DEVICES='${gpu_ids}' MICRO_BATCH_SIZE=256 bash '${launcher}'"
    log "Launched ${session_name} on physical GPUs ${gpu_ids}."
}

read -r delay_seconds target_time < <(seconds_until_next_la_21)
log "Server-side timer armed for ${target_time} (${delay_seconds}s from now)."
sleep "${delay_seconds}"
log "LA 21:00 reached; checking each GPU group before launch."

launch_when_free \
    rdd_unified_l_100300600 \
    0,1,2,3 \
    "${REPO_ROOT}/2d/train_RDD_unified_L_100_300_600_4gpu.sh" &
pid_100=$!

launch_when_free \
    rdd_unified_l_600300100 \
    4,5,6,7 \
    "${REPO_ROOT}/2d/train_RDD_unified_L_600_300_100_4gpu.sh" &
pid_600=$!

wait "${pid_100}"
wait "${pid_600}"
log "Both scheduled launch requests completed."
