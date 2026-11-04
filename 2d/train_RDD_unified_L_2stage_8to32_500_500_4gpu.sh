#!/usr/bin/env bash
# Unified DiT-L/2 two-stage allocation ablation on original ImageNet.
#
# Configuration is written in forward-process order (finest -> coarsest):
#   32x32: 500 timesteps, projection scale 4, then 8x8: 500 timesteps.
# Sampling therefore runs 8x8 -> 32x32 with no 16x16 stage.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RDD_UNIFIED_MODEL="DiT-L/2"
export RDD_UNIFIED_RESOLUTIONS="32,8"
export RDD_UNIFIED_STAGE_LENGTHS="500,500"
export RDD_UNIFIED_RUN_NAME="RDD-Unified-L-2Stage-8to32-500-500"
export RDD_UNIFIED_DEFAULT_GPUS="0,1,2,3"
export RDD_UNIFIED_DEFAULT_MASTER_PORT="29661"
export RDD_UNIFIED_LOG_STEM="train_rdd_unified_l_2stage_8to32_500_500_4gpu"
export DISABLE_SCHEDULED_STOP="1"
export CKPT_EVERY="50000"

exec bash "${SCRIPT_DIR}/train_RDD_unified_L_boundary_4gpu_sf7.sh"
