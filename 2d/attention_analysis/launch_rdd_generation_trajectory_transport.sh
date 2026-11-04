#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT_TAG="${OUT_TAG:-rdd_generation_trajectory_transport}"
OUT="$ROOT/eval_results/$OUT_TAG"
mkdir -p "$OUT"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
.venv-rdd-eval/bin/python -u \
  2d/attention_analysis/probe_rdd_generation_trajectory_transport.py \
  --rdd-ckpt checkpoints/RDD-Unified-XL/0400000.pt \
  --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
  --imagenet-train imagenet/ILSVRC/Data/CLS-LOC/train \
  --output "$OUT/metrics.json" \
  --seed "${SEED:-91001}" \
  2>&1 | tee "$OUT/run.log"
