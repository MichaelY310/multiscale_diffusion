#!/usr/bin/env bash
set -euo pipefail

# Frozen RDD-Unified-L/2 @ 400K (300/300/400) + trainable rank-64 CSRA.
# Four GPUs x 64 images = global batch 256 in one forward/backward; no
# gradient accumulation is used.
ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RDD_CKPT="${RDD_CKPT:-${ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
export RDD_CSRA_MODEL="${RDD_CSRA_MODEL:-DiT-L/2}"
export RDD_CSRA_OUT="${RDD_CSRA_OUT:-${ROOT}/checkpoints/RDD-CSRA-Unified-L-400K-r64-layers1_2_3_14_17_20}"
export RDD_CSRA_NPROC=4
export RDD_CSRA_GLOBAL_BATCH=256
export RDD_CSRA_MICRO_BATCH=256
export RDD_CSRA_MAX_STEPS="${RDD_CSRA_MAX_STEPS:-5000}"
export RDD_CSRA_CHECKPOINT_EVERY="${RDD_CSRA_CHECKPOINT_EVERY:-1000}"

exec bash "${ROOT}/2d/exp/train_rdd_csra_unified_xl_400k_4gpu.sh"
