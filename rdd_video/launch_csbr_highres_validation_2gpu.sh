#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_highres_validation.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_highres_validation_20260910"
NAMES="rdd_vsa20_baseline,uniform6_global075,fine_safe6,combined_disjoint_m2_f6"

cd "${ROOT}"
"${PYTHON}" rdd_video/build_csbr_stage_specific_validation_configs.py
"${PYTHON}" rdd_video/build_csbr_highres_validation_configs.py --output "${SUITE}"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A polar bear walks across floating sea ice beneath the northern lights, slow cinematic tracking shot."
  "A steam locomotive crosses a stone viaduct above a forest valley in autumn, smooth aerial camera movement."
)
seeds=(20261121 20261122)

for index in 0 1; do
  gpu="${index}"
  session="rdd_csbr_highres_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --decode-names '${NAMES}' --latent-only-names '${NAMES}' --prompt '${prompts[$index]}' --size 1280x768 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 3 --timed-rounds 9 --output-dir '${case_dir}' --master-port '$((30641 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
