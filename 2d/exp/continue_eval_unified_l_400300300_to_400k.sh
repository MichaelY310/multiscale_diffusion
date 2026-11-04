#!/usr/bin/env bash
# Continue the Unified-L 400/300/300 convergence evaluation as training emits
# 350K and 400K. Existing complete metrics are detected and never resampled.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
CKPT_DIR="${REPO_ROOT}/checkpoints/RDD-Unified-L-400-300-300/imagenet256-unified/checkpoints"
EVAL_SCRIPT="${REPO_ROOT}/2d/exp/eval_unified_l_400300300_available_4gpu.sh"
TRAIN_SESSION="rdd_unified_l_400300300"
POLL_SECONDS="${POLL_SECONDS:-60}"

wait_for_stable_checkpoint() {
    local step="$1" checkpoint size_before size_after
    checkpoint="${CKPT_DIR}/$(printf '%07d.pt' "${step}")"
    echo "Waiting for checkpoint: ${checkpoint}"
    while [[ ! -s "${checkpoint}" ]]; do
        if ! tmux has-session -t "${TRAIN_SESSION}" 2>/dev/null; then
            echo "ERROR: ${TRAIN_SESSION} exited before ${checkpoint} appeared" >&2
            exit 1
        fi
        sleep "${POLL_SECONDS}"
    done

    # Avoid loading a multi-GB checkpoint while torch.save is still writing it.
    while true; do
        size_before="$(stat -c '%s' "${checkpoint}")"
        sleep "${POLL_SECONDS}"
        size_after="$(stat -c '%s' "${checkpoint}")"
        if [[ "${size_before}" -eq "${size_after}" && "${size_after}" -gt 7000000000 ]]; then
            break
        fi
        echo "Checkpoint still changing: ${size_before} -> ${size_after} bytes"
    done
    echo "Checkpoint stable: ${checkpoint} (${size_after} bytes)"
}

run_available_curve() {
    local requested="$1"
    GPU_IDS="0,1,2,3" WORLD_SIZE="4" REQUESTED_STEPS_CSV="${requested}" \
        bash "${EVAL_SCRIPT}"
}

wait_for_stable_checkpoint 350000
run_available_curve "50000,100000,150000,200000,250000,300000,350000"

wait_for_stable_checkpoint 400000
run_available_curve "50000,100000,150000,200000,250000,300000,350000,400000"

echo "Unified-L 400/300/300 evaluation is complete through 400K."
