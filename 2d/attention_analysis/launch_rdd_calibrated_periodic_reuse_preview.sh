#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT_TAG="${OUT_TAG:-rdd_calibrated_periodic_reuse_preview_400k}"
OUT="$ROOT/eval_results/$OUT_TAG"
mkdir -p "$OUT"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
.venv-rdd-eval/bin/python -u \
  2d/attention_analysis/sample_rdd_calibrated_periodic_reuse_preview.py \
  --rdd-ckpt checkpoints/RDD-Unified-XL/0400000.pt \
  --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
  --imagenet-train imagenet/ILSVRC/Data/CLS-LOC/train \
  --output-dir "$OUT" \
  --seed "${SEED:-20260910}" \
  --benchmark-repeats "${BENCHMARK_REPEATS:-0}" \
  2>&1 | tee "$OUT/run.log"
