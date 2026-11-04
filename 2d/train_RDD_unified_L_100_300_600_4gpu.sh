#!/usr/bin/env bash
# Unified DiT-L/2 allocation ablation: 32/16/8 = 100/300/600.
set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RDD_UNIFIED_STAGE_LENGTHS="100,300,600"
export RDD_UNIFIED_RUN_NAME="RDD-Unified-L-100-300-600"
export RDD_UNIFIED_DEFAULT_GPUS="0,1,2,3"
export RDD_UNIFIED_DEFAULT_MASTER_PORT="29631"
export RDD_UNIFIED_LOG_STEM="train_rdd_unified_l_100_300_600_4gpu"
exec bash "${SCRIPT_DIR}/train_RDD_unified_L_boundary_4gpu_sf7.sh"
