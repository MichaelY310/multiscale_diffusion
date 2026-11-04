#!/usr/bin/env bash
# FastWan official 3-step checkpoint: fixed-resolution baseline vs faux-RDD.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
FASTWAN="${FASTWAN:-${ROOT}/pretrained_models/FastWan2.1-T2V-1.3B-Diffusers}"
WAN="${WAN:-${ROOT}/pretrained_models/Wan2.1-T2V-1.3B}"
KERNEL="${KERNEL:-${ROOT}/third_party/FastVideo-kernel-ref/fastvideo-kernel}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_fastwan_dmd/study_832x448_f29}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
mkdir -p "${OUT}"
cd "${ROOT}"

run_one() {
  local name="$1" factors="$2" density="$3"
  if [[ -s "${OUT}/${name}.mp4" && -s "${OUT}/${name}.metrics.json" ]]; then
    echo "SKIP ${name}"
    return
  fi
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" FASTVIDEO_VSA_TRITON=1 \
    "${PYTHON}" rdd_video/sample_fastwan_faux_rdd.py \
      --fastwan-dir "${FASTWAN}" --original-wan-dir "${WAN}" \
      --fastvideo-kernel-root "${KERNEL}" \
      --prompt "${PROMPT}" --size 832x448 --frame-num 29 --seed 20260905 \
      --dmd-timesteps 1000,757,522 --stage-factors "${factors}" \
      --route-density "${density}" \
      --output "${OUT}/${name}.mp4" \
      --metrics-json "${OUT}/${name}.metrics.json" \
      --latent-output "${OUT}/${name}.latent.pt" \
      2>&1 | tee "${OUT}/${name}.log"
}

# Official FastWan trajectory: all three DMD calls use the requested output size.
run_one fastwan_fullres_dense "1,1;1,1;1,1" 1.0

# Counterfactual: assume the three calls had been RDD-trained at coarse/mid/fine.
run_one fastwan_faux_rdd_dense "4,4;2,2;1,1" 1.0
run_one fastwan_faux_rdd_route75 "4,4;2,2;1,1" 0.75
run_one fastwan_faux_rdd_route50 "4,4;2,2;1,1" 0.50
