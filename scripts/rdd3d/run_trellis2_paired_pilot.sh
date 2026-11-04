#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if tmux has-session -t rdd3d_pilot 2>/dev/null; then
    echo "rdd3d_pilot data-preparation session is still running; wait for it first." >&2
    exit 2
fi

export RDD3D_NUM_GPUS=2
export RDD3D_MODEL_RESOLUTION="${RDD3D_MODEL_RESOLUTION:-512}"
export RDD3D_RUN_NAME="${RDD3D_RUN_NAME:-pilot}"
RDD3D_TRAIN_GPUS="${RDD3D_RDD_GPUS:-4,5}" \
    bash "${ROOT}/scripts/rdd3d/run_trellis2_finetune_tmux.sh" rdd
RDD3D_TRAIN_GPUS="${RDD3D_FLAT_GPUS:-6,7}" \
    bash "${ROOT}/scripts/rdd3d/run_trellis2_finetune_tmux.sh" flat

SUFFIX="train"
if [[ "${RDD3D_TRYRUN:-0}" == 1 ]]; then
    SUFFIX="tryrun"
fi
echo "RDD log:  tail -f ${ROOT}/logs/trellis2_rdd_${RDD3D_MODEL_RESOLUTION}_${RDD3D_RUN_NAME}_${SUFFIX}.log"
echo "Flat log: tail -f ${ROOT}/logs/trellis2_flat_${RDD3D_MODEL_RESOLUTION}_${RDD3D_RUN_NAME}_${SUFFIX}.log"
