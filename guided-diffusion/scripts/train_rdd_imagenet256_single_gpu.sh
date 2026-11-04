#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$ROOT_DIR/.cache/huggingface/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$ROOT_DIR/.cache/huggingface/transformers}"

python - <<'INNER_PY'
import os
import sys

# Avoid importing local ./datasets namespace package by mistake.
cwd = os.getcwd()
sys.path = [p for p in sys.path if p not in ("", ".") and os.path.abspath(p or cwd) != cwd]
from datasets import load_dataset
load_dataset("benjamin-paine/imagenet-1k-256x256", split="train")
print("ImageNet-1k-256x256 cached.")
INNER_PY

python scripts/image_train_with_drift_imagenet256.py   --batch_size "${BATCH_SIZE:-2}"   --num_workers "${NUM_WORKERS:-2}"   --total_steps "${TOTAL_STEPS:-1000}"   --log_every "${LOG_EVERY:-10}"   --ckpt_every "${CKPT_EVERY:-200}"   --lr "${LR:-1e-4}"   --results_dir "${RESULTS_DIR:-guided_diffusion/results_imagenet256_single_gpu}"   --num_channels_64 "${NUM_CHANNELS_64:-64}"   --num_channels_128 "${NUM_CHANNELS_128:-128}"   --num_channels_256 "${NUM_CHANNELS_256:-192}"
