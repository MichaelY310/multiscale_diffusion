#!/usr/bin/env bash
# Resume the two active RDD experiments at the next 20:00 in Los Angeles.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
LOG_FILE="${REPO_ROOT}/logs/resume_all_at_la20.log"
TIMEZONE="America/Los_Angeles"

mkdir -p "${REPO_ROOT}/logs"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$1" | tee -a "${LOG_FILE}"
}

read -r delay_seconds target_time < <(
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
)

log "Scheduled both experiments for ${target_time}; sleeping ${delay_seconds}s on the server."
sleep "${delay_seconds}"

gpu_group_is_idle() {
    local gpu_id
    for gpu_id in "$@"; do
        if nvidia-smi -i "${gpu_id}" --query-compute-apps=pid --format=csv,noheader,nounits \
            | grep -Eq '^[[:space:]]*[0-9]+'; then
            return 1
        fi
    done
    return 0
}

wait_for_gpu_group() {
    local label="$1"
    shift
    while ! gpu_group_is_idle "$@"; do
        log "${label} is occupied; leaving existing processes untouched and retrying in 60s."
        sleep 60
    done
}

start_500500() {
    local session="rdd_unified_l_2stage_500500"
    if pgrep -u "${UID}" -f 'train_RDD_imagenet256_unified.py.*--stage-lengths 500,500' >/dev/null; then
        log "500/500 is already running; refusing to launch a duplicate."
        return 0
    fi
    wait_for_gpu_group "GPUs 0-3" 0 1 2 3
    if tmux has-session -t "${session}" 2>/dev/null; then
        tmux kill-session -t "${session}"
    fi
    tmux new-session -d -s "${session}" \
        "cd '${REPO_ROOT}' && bash 2d/train_RDD_unified_L_2stage_500_500_4gpu_la9.sh"
    log "Started 500/500 in tmux ${session}; launcher will auto-resume the largest numeric checkpoint."
}

start_dmd() {
    local session="rdd_unified_l_phased_dmd3_gb256"
    if pgrep -u "${UID}" -f 'train_rdd_phased_dmd3.py' >/dev/null; then
        log "RDD-DMD is already running; refusing to launch a duplicate."
        return 0
    fi
    wait_for_gpu_group "GPUs 4-7" 4 5 6 7
    if tmux has-session -t "${session}" 2>/dev/null; then
        tmux kill-session -t "${session}"
    fi
    tmux new-session -d -s "${session}" \
        "cd '${REPO_ROOT}' && bash 2d/exp/run_rdd_unified_l_phased_dmd3_dmdpaper_tmux_payload.sh"
    log "Started exact RDD-DMD in tmux ${session}; launcher will auto-resume checkpoints/latest.pt."
}

log "Los Angeles 20:00 reached; checking ownership and GPU availability."
start_500500 &
pid_500500=$!
start_dmd &
pid_dmd=$!
wait "${pid_500500}"
wait "${pid_dmd}"
log "Resume dispatch completed."
