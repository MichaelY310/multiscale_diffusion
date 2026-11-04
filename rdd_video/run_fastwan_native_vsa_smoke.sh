#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/eval_results/rdd_wan_fastwan_native/smoke"
mkdir -p "${OUT}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0 .venv-rdd-video/bin/python rdd_video/sample_fastwan_native_vsa.py \
  --fastvideo-root third_party/FastVideo-ref \
  --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
  --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
  --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
  --prompt "A red fox walks through snow." \
  --size 256x128 --frame-num 9 \
  --vsa-sparsity 0.8 --transition-mode fresh \
  --output "${OUT}/fullres_vsa.mp4" \
  --metrics-json "${OUT}/fullres_vsa.metrics.json" \
  2>&1 | tee "${OUT}/fullres_vsa.log"
