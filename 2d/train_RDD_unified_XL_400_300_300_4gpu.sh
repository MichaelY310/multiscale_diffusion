#!/usr/bin/env bash
# Unified DiT-XL/2 diffusion run matching the legacy FM-v2 stage durations.
#
# Diffusion forward-process order (finest -> coarsest):
#   32x32 / 16x16 / 8x8 = 400 / 300 / 300.
# The legacy RDD-FM Unified-XL-v2 run used chronological FM order
# coarse / middle / fine = 300 / 300 / 400, so these are the same physical
# resolution durations with the time direction reversed.

set -Eeuo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RDD_UNIFIED_MODEL="DiT-XL/2"
export RDD_UNIFIED_RESOLUTIONS="32,16,8"
export RDD_UNIFIED_STAGE_LENGTHS="400,300,300"
export RDD_UNIFIED_RUN_NAME="RDD-Unified-XL-400-300-300"
export RDD_UNIFIED_DEFAULT_GPUS="4,5,6,7"
export RDD_UNIFIED_DEFAULT_MASTER_PORT="29662"
export RDD_UNIFIED_LOG_STEM="train_rdd_unified_xl_400_300_300_4gpu"
export DISABLE_SCHEDULED_STOP="1"
export CKPT_EVERY="50000"

exec bash "${SCRIPT_DIR}/train_RDD_unified_L_boundary_4gpu_sf7.sh"
