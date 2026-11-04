#!/usr/bin/env bash
# Clean FastWan/DMD3 matrix. Every dynamic transition uses canonical noise filling.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_method_matrix/fastwan_f29}"
GPU="${GPU:-0}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
mkdir -p "${OUT}"
cd "${ROOT}"

run_one() {
  local name="$1" factors="$2" sparsity="$3" reuse="$4"
  if [[ -s "${OUT}/${name}.mp4" && -s "${OUT}/${name}.metrics.json" ]]; then
    echo "SKIP ${name}"
    return
  fi
  echo "RUN ${name}: factors=${factors}, VSA sparsity=${sparsity}, reuse=${reuse}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u rdd_video/sample_fastwan_native_vsa.py \
    --fastvideo-root third_party/FastVideo-ref \
    --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
    --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
    --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
    --prompt "${PROMPT}" --size 832x448 --frame-num 29 --seed 20260905 \
    --dmd-timesteps 1000,757,522 --stage-factors "${factors}" \
    --transition-mode fresh --attention-mode vsa \
    --vsa-sparsity "${sparsity}" --reuse-every "${reuse}" \
    --output "${OUT}/${name}.mp4" \
    --metrics-json "${OUT}/${name}.metrics.json" \
    --latent-output "${OUT}/${name}.latent.pt" \
    2>&1 | tee "${OUT}/${name}.log"
}

# Fixed-resolution controls: here StepReuse-Out has 1 or 2 real cache hits.
run_one fixed_vsa20 "1,1;1,1;1,1" 0.8 1
run_one fixed_vsa20_stepreuse2 "1,1;1,1;1,1" 0.8 2
run_one fixed_vsa20_stepreuse3 "1,1;1,1;1,1" 0.8 3

# Full spatiotemporal RDD. "vsa100" retains all fine tile pairs but still
# keeps the checkpoint's learned VSA coarse branch/gate.
run_one rdd_spatiotemporal_vsa100 "4,4;2,2;1,1" 0.0 1
run_one rdd_spatiotemporal_vsa50 "4,4;2,2;1,1" 0.5 1
run_one rdd_spatiotemporal_vsa20 "4,4;2,2;1,1" 0.8 1

# Axis-isolation ablations.
run_one rdd_spatial_only_vsa20 "4,1;2,1;1,1" 0.8 1
run_one rdd_temporal_only_vsa20 "1,4;1,2;1,1" 0.8 1

echo "FASTWAN_MATRIX_COMPLETE"
