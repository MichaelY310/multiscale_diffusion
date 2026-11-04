#!/usr/bin/env bash
# Wait until both corrected Unified-L RDD-FM runs have stable 400K checkpoints,
# then launch their four-GPU FID curves at the same time.  Deliberately does
# not wait for the GPU groups to become idle: the caller has authorized GPU
# sharing with the existing reuse/DMD experiments.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
POLL_SECONDS="${POLL_SECONDS:-30}"
EVAL_SCRIPT="${REPO_ROOT}/2d/exp/eval_rdd_fm_unified_l_convergence_4gpu.sh"
PLOT_WAITER="${REPO_ROOT}/2d/exp/wait_and_plot_rdd_fm_unified_l_curves.sh"
LOG_FILE="${REPO_ROOT}/logs/launch_rdd_fm_unified_l_dual_fid_after_400k.log"

CKPT_A="${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-300-300-400/imagenet256-fm-unified/checkpoints/0400000.pt"
CKPT_B="${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-400-300-300/imagenet256-fm-unified/checkpoints/0400000.pt"
RESULT_DIR_B="${REPO_ROOT}/checkpoints/RDD-FM-Unified-L-corrected-400-300-300"

SESSION_A="eval_rdd_fm_l_300300400_curve"
SESSION_B="eval_rdd_fm_l_400300300_curve"
SESSION_PLOT="plot_rdd_fm_l_allocation_curves"

mkdir -p "${REPO_ROOT}/logs"
[[ -s "${EVAL_SCRIPT}" ]] || { echo "ERROR: missing ${EVAL_SCRIPT}" >&2; exit 1; }
[[ -s "${PLOT_WAITER}" ]] || { echo "ERROR: missing ${PLOT_WAITER}" >&2; exit 1; }

log() {
    printf '[%s] %s\n' "$(date -Is)" "$*" | tee -a "${LOG_FILE}"
}

checkpoints_stable() {
    [[ -s "${CKPT_A}" && -s "${CKPT_B}" ]] || return 1
    local a_before b_before a_after b_after
    a_before="$(stat -c %s "${CKPT_A}")"
    b_before="$(stat -c %s "${CKPT_B}")"
    sleep 15
    a_after="$(stat -c %s "${CKPT_A}")"
    b_after="$(stat -c %s "${CKPT_B}")"
    [[ "${a_before}" == "${a_after}" && "${b_before}" == "${b_after}" ]] || return 1
    (( a_after > 1000000000 && b_after > 1000000000 ))
}

second_trainer_alive() {
    ps -u "$(id -u)" -o args= \
        | grep -F -- "--results-dir ${RESULT_DIR_B}" \
        | grep -v grep >/dev/null
}

log "Waiting for both stable 400K checkpoints. GPU sharing is enabled."
until checkpoints_stable; do
    sleep "${POLL_SECONDS}"
done
log "Both 400K checkpoints are stable."

# Avoid the brief high-memory overlap with the final torch.save/DDP shutdown.
# Other experiments on the same GPUs are intentionally allowed to continue.
while second_trainer_alive; do
    log "400/300/300 checkpoint is ready; waiting for its trainer to exit cleanly."
    sleep 10
done

for session in "${SESSION_A}" "${SESSION_B}" "${SESSION_PLOT}"; do
    if tmux has-session -t "${session}" 2>/dev/null; then
        log "ERROR: tmux session already exists: ${session}; refusing duplicate launch."
        exit 1
    fi
done

tmux new-session -d -s "${SESSION_A}" \
    "cd '${REPO_ROOT}' && env DDPM_ALLOCATION='300,300,400' GPU_IDS='0,1,2,3' bash '${EVAL_SCRIPT}' 2>&1 | tee -a '${REPO_ROOT}/logs/eval_rdd_fm_unified_l_300300400_curve.log'"

tmux new-session -d -s "${SESSION_B}" \
    "cd '${REPO_ROOT}' && env DDPM_ALLOCATION='400,300,300' GPU_IDS='4,5,6,7' bash '${EVAL_SCRIPT}' 2>&1 | tee -a '${REPO_ROOT}/logs/eval_rdd_fm_unified_l_400300300_curve.log'"

tmux new-session -d -s "${SESSION_PLOT}" \
    "cd '${REPO_ROOT}' && bash '${PLOT_WAITER}'"

log "Launched both four-GPU evaluations simultaneously and armed the combined plotter."
log "Sessions: ${SESSION_A}, ${SESSION_B}, ${SESSION_PLOT}"

