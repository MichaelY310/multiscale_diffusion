#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export RDD3D_DYNAMIC_SESSION="${RDD3D_DYNAMIC_SESSION:-rdd3d_dynamic_sparse_gate_2gpu}"
export RDD3D_DYNAMIC_GPU="${RDD3D_DYNAMIC_GPU:-0,1}"
export RDD3D_DYNAMIC_NUM_GPUS="${RDD3D_DYNAMIC_NUM_GPUS:-2}"
export RDD3D_DYNAMIC_RUN_NAME="${RDD3D_DYNAMIC_RUN_NAME:-dynamic_sparse_32asset_gate_w384d12_2gpu}"
export RDD3D_DYNAMIC_MAX_ASSETS="${RDD3D_DYNAMIC_MAX_ASSETS:-32}"
export RDD3D_DYNAMIC_TRAIN_STEPS="${RDD3D_DYNAMIC_TRAIN_STEPS:-10000}"
export RDD3D_DYNAMIC_WIDTH="${RDD3D_DYNAMIC_WIDTH:-384}"
export RDD3D_DYNAMIC_DEPTH="${RDD3D_DYNAMIC_DEPTH:-12}"
export RDD3D_DYNAMIC_WEIGHTS="${RDD3D_DYNAMIC_WEIGHTS:-model}"
export RDD3D_DYNAMIC_SAVE_EVERY="${RDD3D_DYNAMIC_SAVE_EVERY:-1000}"

cd "${ROOT}"
bash scripts/rdd3d/run_dynamic_sparse_pilot_tmux.sh
