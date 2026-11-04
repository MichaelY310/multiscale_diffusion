#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

# Single-GPU setup.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# Keep HuggingFace cache and temp files off /home to avoid disk pressure.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export HF_HOME="${HF_HOME:-/tmp/hf_home}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf_datasets_cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/tmp/hf_transformers_cache}"
export TMPDIR="${TMPDIR:-/tmp}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

mkdir -p \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$MPLCONFIGDIR"

# Shared run defaults.
export BATCH_SIZE="${BATCH_SIZE:-32}"
export BATCH_SIZE_256="${BATCH_SIZE_256:-32}"
export NUM_WORKERS="${NUM_WORKERS:-8}"

# Diffusion defaults.
export TOTAL_STEPS="${TOTAL_STEPS:-1000000}"
export LOG_EVERY="${LOG_EVERY:-100}"
export CKPT_EVERY="${CKPT_EVERY:-5000}"
export DIFFUSION_LR="${DIFFUSION_LR:-1e-4}"
export NUM_CHANNELS_64="${NUM_CHANNELS_64:-64}"
export NUM_CHANNELS_128="${NUM_CHANNELS_128:-128}"
export NUM_CHANNELS_256="${NUM_CHANNELS_256:-192}"
export RESULTS_DIR="${RESULTS_DIR:-guided_diffusion/results_imagenet256_gpu0_bs32}"

# Classifier defaults.
export ITERATIONS="${ITERATIONS:-300000}"
export MICROBATCH="${MICROBATCH:--1}"
export LOG_INTERVAL="${LOG_INTERVAL:-100}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export CLASSIFIER_LR="${CLASSIFIER_LR:-3e-4}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
export OUT_DIR="${OUT_DIR:-guided_diffusion/results_rdd_classifier_gpu0_bs32}"

echo "[1/2] Train RDD diffusion on GPU ${CUDA_VISIBLE_DEVICES} with batch_size=${BATCH_SIZE}"
export LR="$DIFFUSION_LR"
bash scripts/train_rdd_imagenet256_single_gpu.sh

echo "[2/2] Train stage-specific RDD classifiers on GPU ${CUDA_VISIBLE_DEVICES} with batch_size=${BATCH_SIZE}"
export LR="$CLASSIFIER_LR"
bash scripts/train_rdd_classifiers_single_gpu.sh

echo "All done."
