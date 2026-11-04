#!/usr/bin/env bash
# Isolate why the zero-finetune FastWan -> faux-RDD counterfactual collapses.
# This is deliberately small: one prompt, one seed, six targeted controls.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_fastwan_native/root_cause_f29}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
mkdir -p "${OUT}"
cd "${ROOT}"

run_one() {
  local name="$1" factors="$2" transition="$3" sparsity="${4:-0.8}"
  if [[ -s "${OUT}/${name}.mp4" && -s "${OUT}/${name}.metrics.json" ]]; then
    echo "SKIP ${name}"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" rdd_video/sample_fastwan_native_vsa.py \
    --fastvideo-root third_party/FastVideo-ref \
    --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
    --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
    --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
    --prompt "${PROMPT}" --size 832x448 --frame-num 29 --seed 20260905 \
    --dmd-timesteps 1000,757,522 --stage-factors "${factors}" \
    --transition-mode "${transition}" --vsa-sparsity "${sparsity}" \
    --output "${OUT}/${name}.mp4" \
    --metrics-json "${OUT}/${name}.metrics.json" \
    --latent-output "${OUT}/${name}.latent.pt" \
    2>&1 | tee "${OUT}/${name}.log"
}

# Canonical RDD noise filling, matching the established 2-D evaluator:
# upsample clean and draw fresh full-space Gaussian noise.
run_one spatial_gradual_4x2x1_noise_filling "4,1;2,1;1,1" noise_filling
run_one temporal_gradual_4x2x1_noise_filling "1,4;1,2;1,1" noise_filling
run_one spatial_mild_2x1x1_noise_filling "2,1;1,1;1,1" noise_filling

# Optional coupling ablation only: preserve the preceding coarse innovation
# and fill just the fine null space.  This is not the default RDD setting.
run_one full_spatiotemporal_4x2x1_innovation_preserving "4,4;2,2;1,1" innovation_preserving
run_one spatial_gradual_4x2x1_innovation_preserving "4,1;2,1;1,1" innovation_preserving

# VSA at very short coarse sequences is a separate possible failure source.
# Sparsity=0 keeps all fine attention pairs while retaining the checkpoint's
# native VSA block/gates, so only route pruning is removed.
run_one full_spatiotemporal_4x2x1_keep100_noise_filling "4,4;2,2;1,1" noise_filling 0.0

"${PYTHON}" rdd_video/extract_video_middle_frames.py "${OUT}"
