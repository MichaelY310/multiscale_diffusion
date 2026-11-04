#!/usr/bin/env bash
set -euo pipefail

# Same adapter/layers/rank/batch as RDD-CSRA, but the source residual comes
# from the previous full-resolution DiT evaluation (t+4).  This is the matched
# generic-cache control; checkpoint provenance must be recorded separately.
ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
DIT_CKPT="${DIT_L_CKPT:-${ROOT}/checkpoints/DiT-L-2-third-party/DiT-L-2-256x256.pt}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5,6,7}"
export RDD_CKPT="${DIT_CKPT}"
export RDD_CSRA_TEACHER_FAMILY="dit"
export RDD_CSRA_MODEL="DiT-L/2"
export RDD_CSRA_OUT="${RDD_CSRA_OUT:-${ROOT}/checkpoints/DiT-L-TemporalAdapter-r64-layers1_2_3_14_17_20-PRELIMINARY}"
export RDD_CSRA_NPROC=4
export RDD_CSRA_GLOBAL_BATCH=256
export RDD_CSRA_MICRO_BATCH=256
export RDD_CSRA_MAX_STEPS="${RDD_CSRA_MAX_STEPS:-5000}"
export RDD_CSRA_CHECKPOINT_EVERY="${RDD_CSRA_CHECKPOINT_EVERY:-1000}"
export RDD_CSRA_EXTRA_ARGS="--source-gap 4 ${RDD_CSRA_EXTRA_ARGS:-}"

exec bash "${ROOT}/2d/exp/train_rdd_csra_unified_xl_400k_4gpu.sh"
