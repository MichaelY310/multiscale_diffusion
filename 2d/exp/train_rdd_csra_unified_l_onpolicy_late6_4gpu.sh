#!/usr/bin/env bash
set -euo pipefail

# On-policy late-layer CSRA follow-up.  Four GPUs x 64 images = global 256,
# one optimizer step per forward/backward and no gradient accumulation.
ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export RDD_CKPT="${RDD_CKPT:-${ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
export RDD_CSRA_MODEL="DiT-L/2"
export RDD_CSRA_LAYERS="12,14,16,18,20,22"
export RDD_CSRA_TRAINING_MODE="on_policy_layerwise"
export RDD_CSRA_OUT="${RDD_CSRA_OUT:-${ROOT}/checkpoints/RDD-CSRA-OnPolicy-Unified-L-400K-r64-late6}"
export RDD_CSRA_NPROC=4
export RDD_CSRA_GLOBAL_BATCH=256
export RDD_CSRA_MICRO_BATCH=256
export RDD_CSRA_MAX_STEPS="${RDD_CSRA_MAX_STEPS:-3000}"
export RDD_CSRA_CHECKPOINT_EVERY="${RDD_CSRA_CHECKPOINT_EVERY:-500}"

exec bash "${ROOT}/2d/exp/train_rdd_csra_unified_xl_400k_4gpu.sh"
