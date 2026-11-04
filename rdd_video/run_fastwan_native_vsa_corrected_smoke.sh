#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/eval_results/rdd_wan_fastwan_native/corrected_smoke_f29"
mkdir -p "${OUT}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES=0 .venv-rdd-video/bin/python rdd_video/sample_fastwan_native_vsa.py \
  --fastvideo-root third_party/FastVideo-ref \
  --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
  --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
  --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
  --prompt "A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement." \
  --size 832x448 --frame-num 29 --seed 20260905 \
  --stage-factors "1,1;1,1;1,1" --vsa-sparsity 0.8 \
  --transition-mode fresh \
  --output "${OUT}/official_fixed_832x448_vsa20.mp4" \
  --metrics-json "${OUT}/official_fixed_832x448_vsa20.metrics.json" \
  2>&1 | tee "${OUT}/official_fixed_832x448_vsa20.log"
