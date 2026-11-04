#!/usr/bin/env bash
# Unified DiT-L/2 allocation ablation: 32/16/8 = 600/300/100.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RDD_UNIFIED_STAGE_LENGTHS="600,300,100"
export RDD_UNIFIED_RUN_NAME="RDD-Unified-L-600-300-100"
export RDD_UNIFIED_DEFAULT_GPUS="4,5,6,7"
export RDD_UNIFIED_DEFAULT_MASTER_PORT="29632"
export RDD_UNIFIED_LOG_STEM="train_rdd_unified_l_600_300_100_4gpu"
exec bash "${SCRIPT_DIR}/train_RDD_unified_L_boundary_4gpu_sf7.sh"
