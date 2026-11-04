#!/usr/bin/env bash
# Wait for the final checkpoint and an entirely free four-GPU group, then run
# the corresponding corrected RDD-FM Unified-L convergence evaluation.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
DDPM_ALLOCATION="${DDPM_ALLOCATION:?missing DDPM_ALLOCATION}"
GPU_IDS="${GPU_IDS:?missing GPU_IDS}"
POLL_SECONDS="${POLL_SECONDS:-60}"
EVAL_SCRIPT="${REPO_ROOT}/2d/exp/eval_rdd_fm_unified_l_convergence_4gpu.sh"

case "${DDPM_ALLOCATION}" in
    300,300,400) RUN_SUFFIX="300-300-400"; RUN_TAG="300300400" ;;
    400,300,300) RUN_SUFFIX="400-300-300"; RUN_TAG="400300300" ;;
    *) echo "ERROR: unsupported DDPM_ALLOCATION=${DDPM_ALLOCATION}" >&2; exit 1 ;;
esac

FINAL_CKPT="${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-${RUN_SUFFIX}/imagenet256-fm-unified/checkpoints/0400000.pt"
LOG_FILE="${REPO_ROOT}/logs/eval_rdd_fm_unified_l_${RUN_TAG}_queue.log"
mkdir -p "${REPO_ROOT}/logs"
[[ -s "${EVAL_SCRIPT}" ]] || { echo "ERROR: missing ${EVAL_SCRIPT}" >&2; exit 1; }

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

gpu_group_is_free() {
    local requested=",${GPU_IDS}," index uuid app_uuid
    declare -A selected=()
    while IFS=',' read -r index uuid; do
        index="${index//[[:space:]]/}"
        uuid="${uuid//[[:space:]]/}"
        [[ "${requested}" == *",${index},"* ]] && selected["${uuid}"]=1
    done < <(nvidia-smi --query-gpu=index,uuid --format=csv,noheader,nounits)
    (( ${#selected[@]} == 4 )) || return 1
    while IFS=',' read -r app_uuid _; do
        app_uuid="${app_uuid//[[:space:]]/}"
        [[ -z "${selected[${app_uuid}]:-}" ]] || return 1
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits 2>/dev/null || true)
    return 0
}

last_status=""
while true; do
    if [[ ! -s "${FINAL_CKPT}" ]]; then
        status="waiting for 0400000.pt"
    elif ! gpu_group_is_free; then
        status="0400000.pt ready; waiting for GPUs ${GPU_IDS} to become fully free"
    else
        # Make sure a just-written multi-GB checkpoint is stable before load.
        size_before="$(stat -c %s "${FINAL_CKPT}")"
        sleep 15
        size_after="$(stat -c %s "${FINAL_CKPT}")"
        if [[ "${size_before}" != "${size_after}" || "${size_after}" -lt 1000000000 ]]; then
            status="0400000.pt is still being written"
        else
            break
        fi
    fi
    if [[ "${status}" != "${last_status}" ]]; then
        log "${status}"
        last_status="${status}"
    fi
    sleep "${POLL_SECONDS}"
done

log "checkpoint stable and GPUs ${GPU_IDS} free; starting evaluation"
exec env \
    REPO_ROOT="${REPO_ROOT}" \
    DDPM_ALLOCATION="${DDPM_ALLOCATION}" \
    GPU_IDS="${GPU_IDS}" \
    bash "${EVAL_SCRIPT}" 2>&1 | tee -a "${LOG_FILE}"

