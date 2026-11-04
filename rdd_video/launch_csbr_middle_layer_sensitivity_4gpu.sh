#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_middle_layer_sensitivity.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_middle_layer_sensitivity_20260910"
LATENTS="rdd_vsa20_baseline"
for layer in $(seq -w 0 29); do
  LATENTS+=",single_s1_l${layer}_scalar075"
done

cd "${ROOT}"
"${PYTHON}" rdd_video/build_csbr_middle_layer_sensitivity_configs.py --output "${SUITE}"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A blue train crosses a snowy mountain bridge while steam drifts through the valley, cinematic aerial shot."
  "A sea turtle swims through shafts of sunlight above a tropical coral reef, smooth underwater tracking shot."
  "A ceramic artist shapes a clay vase on a spinning wheel in a sunlit studio, close camera movement."
  "A futuristic motorcycle races through a neon city at night while rain reflects on the road, tracking shot."
)
seeds=(20261111 20261112 20261113 20261114)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_middle_sensitivity_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --latent-only-names '${LATENTS}' --prompt '${prompts[$index]}' --size 832x448 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 1 --timed-rounds 1 --output-dir '${case_dir}' --master-port '$((30601 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
