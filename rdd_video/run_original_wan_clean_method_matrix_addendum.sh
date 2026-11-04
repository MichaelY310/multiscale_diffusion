#!/usr/bin/env bash
# Small addendum: valid fixed-resolution controls and C2F density sweep.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_wan_method_matrix/original_wan_f17}"
GPU="${GPU:-1}"
PROMPT="${PROMPT:-A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement.}"
PRETRAINED="${ROOT}/pretrained_models/Wan2.1-T2V-1.3B"
KERNEL_ROOT="${ROOT}/third_party/FastVideo-kernel-ref/fastvideo-kernel"
mkdir -p "${OUT}"
cd "${ROOT}"

run_one() {
  local name="$1" factors="$2" lengths="$3" steps="$4" drift="$5" attention="$6" density="$7" reuse="$8"
  if [[ -s "${OUT}/${name}.mp4" && -s "${OUT}/${name}.metrics.json" ]]; then
    echo "SKIP ${name}"
    return
  fi
  local extra=()
  if [[ "${attention}" == "route" || "${attention}" == "route_reuse" ]]; then
    extra+=(--route-density "${density}" --fastvideo-kernel-root "${KERNEL_ROOT}")
  fi
  if [[ "${attention}" == "reuse" || "${attention}" == "route_reuse" ]]; then
    extra+=(--reuse-every "${reuse}")
  fi
  echo "RUN ${name}: factors=${factors}, attention=${attention}, density=${density}, reuse=${reuse}"
  CUDA_VISIBLE_DEVICES="${GPU}" FASTVIDEO_VSA_TRITON=1 "${PYTHON}" -u \
    rdd_video/sample_rdd_wan.py \
    --use-pretrained-wan --pretrained-dir "${PRETRAINED}" \
    --prompt "${PROMPT}" --size 832x448 --frame-num 17 --seed 20260905 \
    --guidance-scale 5 --solver ab2 --shift 1 \
    --stage-configs "${factors}" --stage-lengths "${lengths}" \
    --drift-scale "${drift}" --steps-per-stage "${steps}" \
    --attention-mode "${attention}" "${extra[@]}" \
    --output "${OUT}/${name}.mp4" \
    --metrics-json "${OUT}/${name}.metrics.json" \
    --latent-output "${OUT}/${name}.latent.pt" \
    2>&1 | tee "${OUT}/${name}.log"
}

# Valid original-Wan controls (no dynamic boundary in these two rows).
run_one fixed_fullres_dense "1,1" "1000" "15" 0 dense 1.0 1
run_one fixed_fullres_stepreuse2 "1,1" "1000" "15" 0 reuse 1.0 2

# Same RDD path; only requested C2F density and reuse change.
run_one rdd_spatiotemporal_c2f75 "4,4;2,2;1,1" "400,300,300" "6,6,3" 1 route 0.75 1
run_one rdd_spatiotemporal_c2f75_stepreuse2 "4,4;2,2;1,1" "400,300,300" "6,6,3" 1 route_reuse 0.75 2
run_one rdd_spatiotemporal_c2f25 "4,4;2,2;1,1" "400,300,300" "6,6,3" 1 route 0.25 1

echo "ORIGINAL_WAN_ADDENDUM_COMPLETE"
