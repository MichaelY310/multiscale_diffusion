#!/usr/bin/env bash
# Run the ten Unified-L convergence evaluations on GPUs 0-7.  Only after all
# five ADM metrics exist for every config, resume both training jobs.  The
# training jobs receive one explicit stop timestamp: tomorrow at 09:00 in
# America/Los_Angeles.  If evaluation reaches/passes that deadline, sampling
# and metric computation are allowed to finish and training is not restarted.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
EVAL_SCRIPT="${REPO_ROOT}/2d/exp/eval_unified_l_boundaries_convergence_50k_250k_8gpu.sh"
ORCH_LOG="${REPO_ROOT}/logs/unified_l_eval_then_resume_la9.log"
CONTROL_FILE="${REPO_ROOT}/logs/unified_l_eval_then_resume_la9.control"
TIMEZONE="America/Los_Angeles"
POLL_SECONDS="${POLL_SECONDS:-60}"

mkdir -p "${REPO_ROOT}/logs"

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${ORCH_LOG}"
}

target_epoch=""
target_iso=""
if [[ -s "${CONTROL_FILE}" ]]; then
    target_epoch="$(awk -F= '$1=="target_epoch"{print $2}' "${CONTROL_FILE}")"
    target_iso="$(awk -F= '$1=="target_iso"{sub(/^[^=]*=/, ""); print}' "${CONTROL_FILE}")"
    if [[ ! "${target_epoch}" =~ ^[0-9]+$ ]] || (( target_epoch <= $(date +%s) )); then
        target_epoch=""
        target_iso=""
    fi
fi

if [[ -z "${target_epoch}" ]]; then
    read -r target_epoch target_iso < <(python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = (now + timedelta(days=1)).replace(hour=9, minute=0, second=0, microsecond=0)
print(int(target.timestamp()), target.isoformat())
PY
    )
fi
printf 'target_epoch=%s\ntarget_iso=%s\ntimezone=%s\n' \
    "${target_epoch}" "${target_iso}" "${TIMEZONE}" > "${CONTROL_FILE}"
log "Fixed training stop target: ${target_iso} (epoch ${target_epoch})."

[[ -x "${EVAL_SCRIPT}" || -f "${EVAL_SCRIPT}" ]] || { log "ERROR: missing ${EVAL_SCRIPT}"; exit 1; }

all_gpus_are_free() {
    local app_count
    app_count="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
        | awk 'NF {count++} END {print count+0}')"
    [[ "${app_count}" -eq 0 ]]
}

while ! all_gpus_are_free; do
    log "At least one GPU is occupied; waiting without signaling or sharing with that process."
    sleep "${POLL_SECONDS}"
done

log "Starting ten checkpoint evaluations; the LA09 deadline never signals sampling/evaluation."
bash "${EVAL_SCRIPT}"
log "All evaluations completed successfully."

if (( $(date +%s) >= target_epoch )); then
    log "Evaluation completed after ${target_iso}; leave training stopped as requested."
    exit 0
fi

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
    (( ${#selected_uuids[@]} == 4 )) || return 1

    while IFS=',' read -r app_uuid app_pid; do
        app_uuid="${app_uuid//[[:space:]]/}"
        if [[ -n "${app_uuid}" && -n "${selected_uuids["${app_uuid}"]:-}" ]]; then
            return 1
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
    return 0
}

launch_when_free() {
    local session_name="$1" gpu_ids="$2" launcher="$3"
    while ! gpu_group_is_free "${gpu_ids}"; do
        if (( $(date +%s) >= target_epoch )); then
            log "Deadline reached while GPUs ${gpu_ids} were occupied; do not launch ${session_name}."
            return 0
        fi
        log "GPUs ${gpu_ids} are occupied; waiting without signaling any process."
        sleep "${POLL_SECONDS}"
    done
    if (( $(date +%s) >= target_epoch )); then
        log "Deadline reached before ${session_name} launch; leave training stopped."
        return 0
    fi
    if tmux has-session -t "${session_name}" 2>/dev/null; then
        log "ERROR: session ${session_name} already exists; refusing duplicate training."
        return 1
    fi
    tmux new-session -d -s "${session_name}" \
        "cd '${REPO_ROOT}' && STOP_AT_EPOCH='${target_epoch}' AUTO_RESUME=1 CUDA_VISIBLE_DEVICES='${gpu_ids}' MICRO_BATCH_SIZE=256 bash '${launcher}'"
    log "Launched ${session_name} on GPUs ${gpu_ids}; exact stop=${target_iso}."
}

launch_when_free \
    rdd_unified_l_100300600 0,1,2,3 \
    "${REPO_ROOT}/2d/train_RDD_unified_L_100_300_600_4gpu.sh" &
pid_100=$!
launch_when_free \
    rdd_unified_l_600300100 4,5,6,7 \
    "${REPO_ROOT}/2d/train_RDD_unified_L_600_300_100_4gpu.sh" &
pid_600=$!
wait "${pid_100}"
wait "${pid_600}"
log "Post-evaluation training launch phase completed."
