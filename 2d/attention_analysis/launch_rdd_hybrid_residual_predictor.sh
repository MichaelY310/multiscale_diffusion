#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT="${OUT:-${ROOT}/eval_results/rdd_hybrid_residual_predictor_400k}"
mkdir -p "${OUT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
CALIBRATION_SEEDS="${CALIBRATION_SEEDS:-101,202}"
VALIDATION_SEEDS="${VALIDATION_SEEDS:-1301,2302,3303}"
CALIBRATION_BATCH="${CALIBRATION_BATCH:-8}"
VALIDATION_BATCH="${VALIDATION_BATCH:-4}"
WARMUP="${WARMUP:-5}"
TIMING_ROUNDS="${TIMING_ROUNDS:-25}"
exec "${ROOT}/.venv-rdd-eval/bin/python" -u \
  "${ROOT}/2d/attention_analysis/probe_rdd_hybrid_residual_predictor.py" \
  --rdd-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
  --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
  --imagenet-train "${ROOT}/imagenet/ILSVRC/Data/CLS-LOC/train" \
  --output "${OUT}/metrics.json" \
  --device cuda:0 \
  --calibration-seeds "${CALIBRATION_SEEDS}" \
  --validation-seeds "${VALIDATION_SEEDS}" \
  --calibration-batch "${CALIBRATION_BATCH}" \
  --validation-batch "${VALIDATION_BATCH}" \
  --warmup "${WARMUP}" \
  --timing-rounds "${TIMING_ROUNDS}"
