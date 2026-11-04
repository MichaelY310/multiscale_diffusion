#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_precompose_validation.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_precompose_validation_20260910"
DECODE="rdd_vsa20_baseline,safe6_multiscale,safe6_multiscale_precompose,safe8_multiscale,safe8_multiscale_precompose"
LATENTS="rdd_vsa20_baseline,safe6_scalar,safe6_multiscale,safe6_multiscale_precompose,safe8_multiscale,safe8_multiscale_precompose"

cd "${ROOT}"
"${PYTHON}" rdd_video/build_csbr_precompose_validation_configs.py
mkdir -p "${OUT_ROOT}"

prompts=(
  "A red panda climbs through a misty bamboo forest while the camera follows from below, cinematic natural light."
  "An old sailing ship crosses rough ocean waves beneath a lightning storm, dramatic tracking shot."
  "A chef decorates a layered cake in a bright bakery, smooth close-up camera movement."
  "Two astronauts walk across a dusty red planet toward a futuristic habitat, wide cinematic shot."
)
seeds=(20261101 20261102 20261103 20261104)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_precompose_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --decode-names '${DECODE}' --latent-only-names '${LATENTS}' --prompt '${prompts[$index]}' --size 832x448 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 5 --timed-rounds 15 --output-dir '${case_dir}' --master-port '$((30561 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
