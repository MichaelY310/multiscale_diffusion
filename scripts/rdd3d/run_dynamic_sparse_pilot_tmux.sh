#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${RDD3D_DYNAMIC_SESSION:-rdd3d_dynamic_sparse_pilot}"
GPU="${RDD3D_DYNAMIC_GPU:-4}"
LOG="${ROOT}/logs/${SESSION}.log"
mkdir -p "${ROOT}/logs"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

TMUX_ENV=(-e "CUDA_VISIBLE_DEVICES=${GPU}")
for NAME in RDD3D_DYNAMIC_PYTHON RDD3D_DYNAMIC_DATA_ROOT RDD3D_DYNAMIC_LATENTS \
    RDD3D_DYNAMIC_RENDERS RDD3D_DYNAMIC_RUN_NAME RDD3D_DYNAMIC_OUTPUT \
    RDD3D_DYNAMIC_MAX_ASSETS RDD3D_DYNAMIC_TRAIN_STEPS RDD3D_DYNAMIC_WIDTH \
    RDD3D_DYNAMIC_DEPTH RDD3D_DYNAMIC_WEIGHTS RDD3D_DYNAMIC_NUM_GPUS \
    RDD3D_DYNAMIC_SAVE_EVERY; do
    if [[ -n "${!NAME:-}" ]]; then
        TMUX_ENV+=(-e "${NAME}=${!NAME}")
    fi
done

tmux new-session -d -s "${SESSION}" "${TMUX_ENV[@]}" \
    "bash -lc 'cd \"${ROOT}\" && bash scripts/rdd3d/run_dynamic_sparse_pilot.sh 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION} on physical GPU ${GPU}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
