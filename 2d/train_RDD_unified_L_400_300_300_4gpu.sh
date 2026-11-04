#!/usr/bin/env bash
# Unified DiT-L/2 training-allocation ablation on original ImageNet.
#
# Forward-process order is finest -> coarsest:
#   32x32 / 16x16 / 8x8 = 400 / 300 / 300.
# This isolates the allocation change against the existing Unified-L
# 300/300/400, 100/300/600, and 600/300/100 convergence curves.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RDD_UNIFIED_MODEL="DiT-L/2"
export RDD_UNIFIED_RESOLUTIONS="32,16,8"
export RDD_UNIFIED_STAGE_LENGTHS="400,300,300"
export RDD_UNIFIED_RUN_NAME="RDD-Unified-L-400-300-300"
export RDD_UNIFIED_DEFAULT_GPUS="4,5,6,7"
export RDD_UNIFIED_DEFAULT_MASTER_PORT="29662"
export RDD_UNIFIED_LOG_STEM="train_rdd_unified_l_400_300_300_4gpu"
export CKPT_EVERY="50000"
export DISABLE_SCHEDULED_STOP="1"

exec bash "${SCRIPT_DIR}/train_RDD_unified_L_boundary_4gpu_sf7.sh"
