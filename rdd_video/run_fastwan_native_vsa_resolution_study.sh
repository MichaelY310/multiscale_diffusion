#!/usr/bin/env bash
# Official FastWan/VSA fixed-resolution controls plus a faux-RDD counterfactual.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_fastwan_native/study_f29}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
mkdir -p "${OUT}"
cd "${ROOT}"

run_one() {
  local name="$1" size="$2" factors="$3"
  if [[ -s "${OUT}/${name}.mp4" && -s "${OUT}/${name}.metrics.json" ]]; then
    echo "SKIP ${name}"
    return
  fi
  CUDA_VISIBLE_DEVICES=0 "${PYTHON}" rdd_video/sample_fastwan_native_vsa.py \
    --fastvideo-root third_party/FastVideo-ref \
    --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
    --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
    --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
    --prompt "${PROMPT}" --size "${size}" --frame-num 29 --seed 20260905 \
    --dmd-timesteps 1000,757,522 --stage-factors "${factors}" \
    --vsa-sparsity 0.8 --transition-mode fresh \
    --output "${OUT}/${name}.mp4" \
    --metrics-json "${OUT}/${name}.metrics.json" \
    --latent-output "${OUT}/${name}.latent.pt" \
    2>&1 | tee "${OUT}/${name}.log"
}

# FastWan's real capability: choose a fixed grid for the complete trajectory.
run_one official_fixed_832x448_vsa20 832x448 "1,1;1,1;1,1"
run_one official_fixed_416x224_vsa20 416x224 "1,1;1,1;1,1"

# Counterfactual only.  The released FastWan checkpoint is not RDD-trained.
run_one faux_rdd_4x2x1_832x448_vsa20 832x448 "4,4;2,2;1,1"
