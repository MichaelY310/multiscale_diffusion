#!/usr/bin/env bash
# One-shot server-side timer for the two currently active Unified-L ablations.
# At the next 20:00 in America/Los_Angeles it resumes each job from its newest
# numeric checkpoint.  It never signals an existing process: an occupied GPU
# group is polled until it becomes free.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TIMEZONE="America/Los_Angeles"
POLL_SECONDS="${POLL_SECONDS:-60}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/scheduled_resume_active_unified_l_la20.log}"

mkdir -p "${REPO_ROOT}/logs"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

seconds_until_next_la_20() {
    python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=20, minute=0, second=0, microsecond=0)
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
        log "ERROR: expected four GPU UUIDs for physical GPUs ${requested_csv}."
        return 1
    fi

    while IFS=',' read -r app_uuid app_pid; do
        app_uuid="${app_uuid//[[:space:]]/}"
        app_pid="${app_pid//[[:space:]]/}"
        if [[ -n "${selected_uuids["${app_uuid}"]:-}" ]]; then
            log "Physical GPUs ${requested_csv} are occupied by PID ${app_pid}; waiting without signaling it."
            return 1
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
    return 0
}

matching_trainer_is_alive() {
    local run_name="$1"
    ps -u "$(id -u)" -o args= | grep -F -- "--results-dir ${REPO_ROOT}/checkpoints/${run_name}" | grep -v grep >/dev/null
}

launch_when_free() {
    local session_name="$1"
    local gpu_ids="$2"
    local run_name="$3"
    local launcher="$4"
    local tmux_log="$5"

    if matching_trainer_is_alive "${run_name}"; then
        log "${run_name} is already running under this UID; refusing a duplicate launch."
        return 0
    fi

    while ! gpu_group_is_free "${gpu_ids}"; do
        sleep "${POLL_SECONDS}"
    done

    if matching_trainer_is_alive "${run_name}"; then
        log "${run_name} started while GPUs were being polled; refusing a duplicate launch."
        return 0
    fi
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        log "ERROR: tmux session ${session_name} already exists; refusing to replace it."
        return 1
    fi

    tmux new-session -d -s "${session_name}" \
        "cd '${REPO_ROOT}' && AUTO_RESUME=1 CUDA_VISIBLE_DEVICES='${gpu_ids}' MICRO_BATCH_SIZE=256 bash '${launcher}' 2>&1 | tee -a '${tmux_log}'"
    log "Launched ${run_name} in tmux ${session_name} on physical GPUs ${gpu_ids}; AUTO_RESUME=1."
}

read -r delay_seconds target_time < <(seconds_until_next_la_20)
log "One-shot server timer armed for ${target_time} (${delay_seconds}s from now)."
sleep "${delay_seconds}"
log "Los Angeles 20:00 reached; checking both GPU groups."

launch_when_free \
    rdd_unified_l_8to32_500500 \
    0,1,2,3 \
    RDD-Unified-L-2Stage-8to32-500-500 \
    "${REPO_ROOT}/2d/train_RDD_unified_L_2stage_8to32_500_500_4gpu.sh" \
    "${REPO_ROOT}/logs/train_rdd_unified_l_2stage_8to32_500_500_tmux.log" &
pid_500=$!

launch_when_free \
    rdd_unified_l_400300300 \
    4,5,6,7 \
    RDD-Unified-L-400-300-300 \
    "${REPO_ROOT}/2d/train_RDD_unified_L_400_300_300_4gpu.sh" \
    "${REPO_ROOT}/logs/train_rdd_unified_l_400_300_300_tmux.log" &
pid_400=$!

status=0
wait "${pid_500}" || status=1
wait "${pid_400}" || status=1
if (( status == 0 )); then
    log "Both scheduled launch requests completed successfully."
else
    log "ERROR: at least one scheduled launch request failed; inspect this log."
fi
exit "${status}"
