#!/usr/bin/env bash
# Server-side one-shot scheduler for the two Unified-L RDD-FM parity runs.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TIMEZONE="America/Los_Angeles"
TRAINER="${REPO_ROOT}/2d/train_RDD_FM_unified_L_allocation_4gpu.sh"
SCHEDULER_LOG="${REPO_ROOT}/logs/schedule_rdd_fm_unified_l_pair_la19.log"

mkdir -p "${REPO_ROOT}/logs"
[[ -f "${TRAINER}" ]] || { echo "ERROR: missing trainer ${TRAINER}" >&2; exit 1; }

read -r DELAY_SECONDS TARGET_ISO < <(python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=19, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(max(0, int((target - now).total_seconds())), target.isoformat())
PY
)

printf '[%s] Waiting %s seconds for %s\n' "$(date -Is)" "${DELAY_SECONDS}" "${TARGET_ISO}" | tee -a "${SCHEDULER_LOG}"
sleep "${DELAY_SECONDS}"

launch_one() {
    local session="$1" gpus="$2" allocation="$3"
    if tmux has-session -t "${session}" 2>/dev/null; then
        printf '[%s] ERROR: tmux session already exists: %s\n' "$(date -Is)" "${session}" | tee -a "${SCHEDULER_LOG}"
        return 1
    fi
    tmux new-session -d -s "${session}" \
        "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='${gpus}' DDPM_ALLOCATION='${allocation}' bash '${TRAINER}'"
    printf '[%s] Started %s on GPUs %s for allocation %s\n' \
        "$(date -Is)" "${session}" "${gpus}" "${allocation}" | tee -a "${SCHEDULER_LOG}"
}

launch_one rdd_fm_unified_l_300300400 0,1,2,3 300,300,400
launch_one rdd_fm_unified_l_400300300 4,5,6,7 400,300,300

sleep 5
for session in rdd_fm_unified_l_300300400 rdd_fm_unified_l_400300300; do
    if tmux has-session -t "${session}" 2>/dev/null; then
        printf '[%s] Verified running: %s\n' "$(date -Is)" "${session}" | tee -a "${SCHEDULER_LOG}"
    else
        printf '[%s] ERROR: launch did not remain alive: %s\n' "$(date -Is)" "${session}" | tee -a "${SCHEDULER_LOG}"
        exit 1
    fi
done

