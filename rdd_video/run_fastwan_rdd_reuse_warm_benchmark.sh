#!/usr/bin/env bash
# FastWan-only: warm steady-state latency and RDD + StepReuse-Out quality probes.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_method_matrix/fastwan_reuse_warm_f29}"
GPU="${GPU:-0}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
WARMUP_RUNS="${WARMUP_RUNS:-1}"
TIMED_RUNS="${TIMED_RUNS:-3}"
mkdir -p "${OUT}"
cd "${ROOT}"

run_one() {
  local name="$1" factors="$2" reuse="$3"
  if [[ -s "${OUT}/${name}.mp4" && -s "${OUT}/${name}.metrics.json" ]]; then
    echo "SKIP ${name}"
    return
  fi
  echo "RUN ${name}: factors=${factors}, VSA keep=20%, reuse=${reuse}, warmup=${WARMUP_RUNS}, timed=${TIMED_RUNS}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" -u rdd_video/sample_fastwan_native_vsa.py \
    --fastvideo-root third_party/FastVideo-ref \
    --fastvideo-kernel-root third_party/FastVideo-ref/fastvideo-kernel \
    --fastwan-dir pretrained_models/FastWan2.1-T2V-1.3B-Diffusers \
    --original-wan-dir pretrained_models/Wan2.1-T2V-1.3B \
    --prompt "${PROMPT}" --size 832x448 --frame-num 29 --seed 20260905 \
    --dmd-timesteps 1000,757,522 --stage-factors "${factors}" \
    --transition-mode fresh --attention-mode vsa \
    --vsa-sparsity 0.8 --reuse-every "${reuse}" \
    --warmup-runs "${WARMUP_RUNS}" --timed-runs "${TIMED_RUNS}" \
    --output "${OUT}/${name}.mp4" \
    --metrics-json "${OUT}/${name}.metrics.json" \
    --latent-output "${OUT}/${name}.latent.pt" \
    2>&1 | tee "${OUT}/${name}.log"
}

# Released FastWan DMD3 reference and a fixed-resolution reuse control.
run_one fixed_vsa20_hot "1,1;1,1;1,1" 1
run_one fixed_vsa20_reuse2_hot "1,1;1,1;1,1" 2

# Two-stage RDD paths. The first two DMD calls share a coarse shape, so the
# reuse2 rows contain one real StepReuse-Out hit per transformer layer.
run_one rdd_temporal_2stage_hot "1,2;1,2;1,1" 1
run_one rdd_temporal_2stage_reuse2_hot "1,2;1,2;1,1" 2
run_one rdd_spatial_2stage_hot "2,1;2,1;1,1" 1
run_one rdd_spatial_2stage_reuse2_hot "2,1;2,1;1,1" 2
run_one rdd_spatiotemporal_2stage_hot "2,2;2,2;1,1" 1
run_one rdd_spatiotemporal_2stage_reuse2_hot "2,2;2,2;1,1" 2

# Original three-resolution RDD path: no same-shape call, hence no reuse hit.
run_one rdd_spatiotemporal_3stage_hot "4,4;2,2;1,1" 1

echo "FASTWAN_RDD_REUSE_WARM_COMPLETE"
