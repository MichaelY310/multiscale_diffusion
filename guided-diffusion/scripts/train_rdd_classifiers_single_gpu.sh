#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT_DIR/.cache/huggingface/transformers}"

python - <<'INNER_PY'
from datasets import load_dataset
load_dataset("benjamin-paine/imagenet-1k-256x256", split="train")
print("ImageNet-1k-256x256 cached.")
INNER_PY

COMMON_ARGS=(
  --iterations "${ITERATIONS:-5000}"
  --batch_size "${BATCH_SIZE:-4}"
  --microbatch "${MICROBATCH:--1}"
  --num_workers "${NUM_WORKERS:-2}"
  --log_interval "${LOG_INTERVAL:-10}"
  --save_interval "${SAVE_INTERVAL:-500}"
  --lr "${LR:-3e-4}"
  --weight_decay "${WEIGHT_DECAY:-0.05}"
  --out_dir "${OUT_DIR:-guided_diffusion/results_rdd_classifier_single_gpu}"
)

python scripts/classifier_train_with_drift_imagenet256.py --stage 64 "${COMMON_ARGS[@]}"
python scripts/classifier_train_with_drift_imagenet256.py --stage 128 "${COMMON_ARGS[@]}"
python scripts/classifier_train_with_drift_imagenet256.py --stage 256 "${COMMON_ARGS[@]}" --batch_size "${BATCH_SIZE_256:-2}"
