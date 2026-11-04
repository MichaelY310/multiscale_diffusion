#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_method_matrix/smoke}"
GPU="${GPU:-0}"
mkdir -p "${OUT}"
cd "${ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" .venv-rdd-video/bin/python -u \
  rdd_video/sample_fastwan_native_vsa.py \
  --fastvideo-root third_party/FastVideo-ref \
  --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
  --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
  --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
  --prompt "A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement." \
  --size 832x448 --frame-num 5 --seed 20260905 \
  --dmd-timesteps 1000,757,522 \
  --stage-factors "4,4;2,2;1,1" \
  --transition-mode fresh \
  --attention-mode c2f_route --route-density 0.2 --reuse-every 1 \
  --output "${OUT}/c2f20.mp4" \
  --metrics-json "${OUT}/c2f20.metrics.json" \
  --latent-output "${OUT}/c2f20.latent.pt"
