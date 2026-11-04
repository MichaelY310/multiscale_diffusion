#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SESSION="${RDD3D_FORMAL_SESSION:-rdd3d_formal_heldout}"
LOG="${ROOT}/logs/${SESSION}.log"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

TMUX_ENV=()
for NAME in RDD3D_FORMAL_DATA_ROOT RDD3D_FORMAL_LATENTS RDD3D_FORMAL_RENDERS \
    RDD3D_PILOT_LATENTS RDD3D_FORMAL_TRAIN_COUNT RDD3D_FORMAL_HELDOUT_COUNT \
    RDD3D_FORMAL_SPLIT_SEED RDD3D_FORMAL_SPLIT_ROOT \
    RDD3D_FORMAL_INIT_CHECKPOINT RDD3D_FORMAL_INIT_STEP \
    RDD3D_FORMAL_TEACHER_STEPS RDD3D_FORMAL_REPLAY_STEPS \
    RDD3D_FORMAL_RUN_NAME RDD3D_FORMAL_MESH_LIMIT RDD3D_DYNAMIC_PYTHON; do
    if [[ -n "${!NAME:-}" ]]; then
        TMUX_ENV+=(-e "${NAME}=${!NAME}")
    fi
done

tmux new-session -d -s "${SESSION}" "${TMUX_ENV[@]}" \
    "bash -lc 'cd \"${ROOT}\" && bash scripts/rdd3d/run_dynamic_sparse_formal_heldout_2gpu.sh 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
