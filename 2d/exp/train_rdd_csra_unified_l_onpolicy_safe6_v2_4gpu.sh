#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
export RDD_ROOT="${ROOT}"
export RDD_CSRA_MODEL="DiT-L/2"
export RDD_CSRA_HIDDEN_SIZE=1024
export RDD_CKPT="${RDD_CKPT:-${ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
export RDD_CSRA_OUT="${RDD_CSRA_OUT:-${ROOT}/checkpoints/RDD-CSRA-OnPolicy-Unified-L-400K-r128-safe6-v2}"
export RDD_CSRA_LAYERS="${RDD_CSRA_LAYERS:-8,11,13,15,17,20}"
export RDD_CSRA_RANK="${RDD_CSRA_RANK:-128}"
export RDD_CSRA_ARCHITECTURE="fine_context_v2"
export RDD_CSRA_TRAINING_MODE="on_policy_layerwise"
export RDD_CSRA_GLOBAL_BATCH=256
export RDD_CSRA_MICRO_BATCH=256
export RDD_CSRA_MAX_STEPS="${RDD_CSRA_MAX_STEPS:-3000}"
export RDD_CSRA_CHECKPOINT_EVERY="${RDD_CSRA_CHECKPOINT_EVERY:-500}"
export RDD_CSRA_NPROC=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

exec bash "${ROOT}/2d/exp/train_rdd_csra_unified_xl_400k_4gpu.sh"

