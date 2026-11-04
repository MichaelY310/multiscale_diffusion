#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT="${OUT:-${ROOT}/eval_results/rdd_cross_stage_timegrid_400k}"
mkdir -p "${OUT}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec "${ROOT}/.venv-rdd-eval/bin/python" -u \
  "${ROOT}/2d/attention_analysis/probe_rdd_cross_stage_timegrid.py" \
  --rdd-ckpt "${ROOT}/checkpoints/RDD-Unified-XL/0400000.pt" \
  --vae-path "${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema" \
  --imagenet-train "${ROOT}/imagenet/ILSVRC/Data/CLS-LOC/train" \
  --output "${OUT}/metrics.json" \
  --device cuda:0 \
  --seed 9191 \
  --batch-size 4
