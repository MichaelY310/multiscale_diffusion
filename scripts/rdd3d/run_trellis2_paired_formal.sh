#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SOURCE="${RDD3D_SOURCE:-sketchfab}"
COUNT="${RDD3D_SUBSET_COUNT:-50000}"

export RDD3D_MODEL_RESOLUTION="${RDD3D_MODEL_RESOLUTION:-1024}"
export RDD3D_DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_${SOURCE}_train_${COUNT}}"
export RDD3D_DATASET_NAME="ObjaverseXL_${SOURCE}_train"
export RDD3D_MAX_STEPS="${RDD3D_MAX_STEPS:-20000}"
export RDD3D_RUN_NAME="${RDD3D_RUN_NAME:-formal_${SOURCE}_${COUNT}_${RDD3D_MAX_STEPS}step}"
export RDD3D_BATCH_PER_GPU="${RDD3D_BATCH_PER_GPU:-1}"
export RDD3D_MIN_AESTHETIC_SCORE="${RDD3D_MIN_AESTHETIC_SCORE:-5.5}"
export RDD3D_IMAGE_ENCODER="${RDD3D_IMAGE_ENCODER:-dinov3}"
export RDD3D_EMA_RATE="${RDD3D_EMA_RATE:-0.999}"
export RDD3D_RDD_GPUS="${RDD3D_RDD_GPUS:-4,5}"
export RDD3D_FLAT_GPUS="${RDD3D_FLAT_GPUS:-6,7}"

if [[ ! -d "${RDD3D_DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${RDD3D_MODEL_RESOLUTION}" ]]; then
    echo "Prepared ${RDD3D_MODEL_RESOLUTION} latents not found under ${RDD3D_DATA_ROOT}." >&2
    echo "Run scripts/rdd3d/run_trellis2_formal_data_tmux.sh first." >&2
    exit 2
fi

bash "${ROOT}/scripts/rdd3d/run_trellis2_paired_pilot.sh"
