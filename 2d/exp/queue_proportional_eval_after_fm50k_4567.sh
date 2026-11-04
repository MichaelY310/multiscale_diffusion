#!/usr/bin/env bash
# Preserve the currently running 4-7 FM job until its first durable 50K
# checkpoint, run the proportional-sampler audit, then resume that exact FM
# allocation.  No process is signalled by PID; only the exact named tmux
# session is stopped, after its command line and checkpoint are verified.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
FM_SESSION="rdd_fm_unified_l_400300300"
FM_RESULTS="${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-400-300-300"
FM_CKPT="${FM_RESULTS}/imagenet256-fm-unified/checkpoints/0050000.pt"
FM_TRAINER="${REPO_ROOT}/2d/train_RDD_FM_unified_L_allocation_4gpu.sh"
EVAL_SCRIPT="${REPO_ROOT}/2d/exp/eval_unified_l_interval_proportional_nfe250_4567.sh"
POLL_SECONDS="${POLL_SECONDS:-60}"
training_stopped=0

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*"
}

session_is_expected_fm_job() {
    local pane_pid sid process_dump
    pane_pid="$(tmux list-panes -t "${FM_SESSION}" -F '#{pane_pid}' 2>/dev/null | head -n 1)"
    [[ "${pane_pid}" =~ ^[0-9]+$ ]] || return 1
    sid="$(ps -o sid= -p "${pane_pid}" | tr -d ' ')"
    [[ "${sid}" =~ ^[0-9]+$ ]] || return 1
    process_dump="$(ps -ww -o user=,pid=,args= --sid "${sid}")"
    grep -Fq "${REPO_ROOT}/2d/train_RDD_FM_imagenet256_unified.py" <<< "${process_dump}" || return 1
    grep -Fq "${FM_RESULTS}" <<< "${process_dump}" || return 1
    if awk -v expected="$(id -un)" '$1 != expected {exit 1}' <<< "${process_dump}"; then
        return 0
    fi
    return 1
}

gpus_are_idle() {
    local gpu pids
    for gpu in 4 5 6 7; do
        pids="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null | tr -cd '0-9\n')"
        [[ -z "${pids}" ]] || return 1
    done
    return 0
}

resume_fm() {
    if (( training_stopped == 0 )); then
        return
    fi
    if tmux has-session -t "${FM_SESSION}" 2>/dev/null; then
        log "FM session already exists; not creating a duplicate."
        return
    fi
    tmux new-session -d -s "${FM_SESSION}" \
        "cd '${REPO_ROOT}' && CUDA_VISIBLE_DEVICES='4,5,6,7' DDPM_ALLOCATION='400,300,300' bash '${FM_TRAINER}'"
    log "Resumed ${FM_SESSION} on GPUs 4,5,6,7 from the latest checkpoint."
    training_stopped=0
}

trap resume_fm EXIT

for required in "${FM_TRAINER}" "${EVAL_SCRIPT}"; do
    [[ -s "${required}" ]] || { log "ERROR: missing ${required}"; exit 1; }
done

log "Waiting for atomic FM checkpoint ${FM_CKPT}."
while [[ ! -s "${FM_CKPT}" ]]; do
    if ! tmux has-session -t "${FM_SESSION}" 2>/dev/null; then
        log "ERROR: ${FM_SESSION} disappeared before the 50K checkpoint was written."
        exit 1
    fi
    sleep "${POLL_SECONDS}"
done

if tmux has-session -t "${FM_SESSION}" 2>/dev/null; then
    if ! session_is_expected_fm_job; then
        log "ERROR: exact-token validation failed; refusing to stop ${FM_SESSION}."
        exit 1
    fi
    log "50K checkpoint verified. Stopping only exact tmux session ${FM_SESSION}."
    tmux kill-session -t "${FM_SESSION}"
    training_stopped=1
fi

log "Waiting until GPUs 4,5,6,7 contain no compute processes."
while ! gpus_are_idle; do
    sleep 10
done

log "Starting interval-proportional Unified-L evaluation on GPUs 4,5,6,7."
GPU_IDS=4,5,6,7 bash "${EVAL_SCRIPT}"
log "Evaluation finished successfully."

resume_fm
trap - EXIT
