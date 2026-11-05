#!/usr/bin/env bash
# Corrected RDD-FM Unified XL/2 paper run.
#
# Target allocation, expressed in resolution order (finest -> coarsest):
#   32x32: 300   16x16: 300   8x8: 400
#
# RDD-FM uses data time (t=0 noise, t=1 data), so the corresponding
# coarse -> middle -> fine intervals are:
#   8x8: [0.0, 0.4]   16x16: [0.4, 0.7]   32x32: [0.7, 1.0]
#
# The current Python entry accepts --stage-lengths in FM data-time order
# (coarsest first), hence the intentionally reversed CLI value 400,300,300.
# This is the strict stage-boundary mirror of DDPM 300,300,400.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
# Global micro batch, not per-GPU.  With two GPUs the default is 4/GPU and
# 32 accumulation rounds, preserving the effective global batch of 256.
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-8}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-400000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MASTER_PORT="${MASTER_PORT:-29621}"

DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/imagenet}"
if [[ -d "${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train" ]]; then
    DATA_PATH="${DATA_ROOT}/ILSVRC/Data/CLS-LOC/train"
elif [[ -d "${DATA_ROOT}/train" ]]; then
    DATA_PATH="${DATA_ROOT}/train"
elif [[ -d "${DATA_ROOT}" ]]; then
    DATA_PATH="${DATA_ROOT}"
else
    echo "ERROR: ImageNet directory not found under ${DATA_ROOT}" >&2
    exit 1
fi

RESULTS_DIR="${RESULTS_DIR:-${REPO_ROOT}/checkpoints/RDD-FM-Unified-XL-corrected-300-300-400}"
TRAIN_ENTRY="${REPO_ROOT}/2d/train_RDD_FM_imagenet256_unified.py"

if [[ ! -f "${TRAIN_ENTRY}" ]]; then
    echo "ERROR: training entry not found: ${TRAIN_ENTRY}" >&2
    exit 1
fi

# Activate the repository environment when present; otherwise use the caller's
# already-active Python environment.
if [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

IFS=',' read -r -a GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
NPROC_PER_NODE="${#GPU_IDS[@]}"
if (( NPROC_PER_NODE < 1 )); then
    echo "ERROR: CUDA_VISIBLE_DEVICES does not contain any GPU id" >&2
    exit 1
fi

if (( GLOBAL_BATCH_SIZE % MICRO_BATCH_SIZE != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE} must be divisible by MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE}" >&2
    exit 1
fi
if (( MICRO_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
    echo "ERROR: MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE} must be divisible by GPU count=${NPROC_PER_NODE}" >&2
    exit 1
fi

echo "============================================================"
echo "RDD-FM Unified XL/2 — corrected DDPM-300/300/400 mirror"
echo "GPUs: ${CUDA_VISIBLE_DEVICES} (${NPROC_PER_NODE} processes)"
echo "Resolution allocation (fine/mid/coarse): 300/300/400"
echo "FM boundaries: [0.0, 0.4, 0.7, 1.0]"
echo "FM intervals (coarse/mid/fine): 400/300/300"
echo "Global batch: ${GLOBAL_BATCH_SIZE}; global micro batch: ${MICRO_BATCH_SIZE}"
echo "Target optimizer step: ${MAX_TRAIN_STEPS}"
echo "ImageNet: ${DATA_PATH}"
echo "Results: ${RESULTS_DIR}"
echo "============================================================"

mkdir -p "${RESULTS_DIR}"

TRAIN_ARGS=(
    --model DiT-XL/2
    --resolutions 32,16,8
    --stage-lengths 400,300,300
    --diffusion-steps 1000
    --results-dir "${RESULTS_DIR}"
    --global-batch-size "${GLOBAL_BATCH_SIZE}"
    --micro-batch-size "${MICRO_BATCH_SIZE}"
    --max-train-steps "${MAX_TRAIN_STEPS}"
    --num-workers "${NUM_WORKERS}"
    --no-pin-memory
    --epochs 1400
    --log-every 100
    --ckpt-every 5000
    --vae ema
    --global-seed 0
    --data-source imagenet
    --data-path "${DATA_PATH}"
)

# Resume only from a checkpoint produced by this corrected run.  The Python
# entry validates saved stage lengths/boundaries and rejects legacy FM-v2.
if [[ -n "${RESUME_CKPT:-}" ]]; then
    TRAIN_ARGS+=(--resume "${RESUME_CKPT}")
fi

cd "${REPO_ROOT}"
torchrun \
    --nnodes=1 \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${TRAIN_ENTRY}" \
    "${TRAIN_ARGS[@]}"
