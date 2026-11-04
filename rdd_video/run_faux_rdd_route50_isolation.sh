#!/usr/bin/env bash
# One extra sample that separates 50% route sparsity from attention-output reuse.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="${REPO_DIR}/eval_results/rdd_wan_faux_sampling/study_832x448_f17"
mkdir -p "${DIR}"
cd "${REPO_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" FASTVIDEO_VSA_TRITON=1 \
  .venv-rdd-video/bin/python rdd_video/sample_rdd_wan.py \
  --use-pretrained-wan \
  --pretrained-dir pretrained_models/Wan2.1-T2V-1.3B \
  --prompt "A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement." \
  --size 832x448 --frame-num 17 --seed 20260905 --guidance-scale 5 \
  --solver ab2 --shift 1 \
  --stage-configs "4,4;2,2;1,1" --stage-lengths "400,300,300" \
  --drift-scale 1 --steps-per-stage "6,6,3" \
  --attention-mode route --route-density 0.50 \
  --fastvideo-kernel-root third_party/FastVideo-kernel-ref/fastvideo-kernel \
  --output "${DIR}/faux_rdd_route50.mp4" \
  --metrics-json "${DIR}/faux_rdd_route50.metrics.json" \
  --latent-output "${DIR}/faux_rdd_route50.latent.pt"
