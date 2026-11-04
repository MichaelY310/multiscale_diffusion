#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_single_layer_sensitivity.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_layer_sensitivity_20260910"
LATENTS="rdd_vsa20_baseline"
for layer in $(seq -w 0 29); do
  LATENTS+=",single_l${layer}_multiscale"
done
mkdir -p "${OUT_ROOT}"

prompts=(
  "A symphony orchestra performs in a grand concert hall as the camera slowly moves between musicians, warm cinematic lighting."
  "A powerful waterfall flows beside an ancient stone castle surrounded by autumn trees, smooth aerial movement."
  "A panda wearing a chef hat kneads dough in a bright bakery while flour floats in the air, close tracking shot."
  "A silver spacecraft flies through colorful clouds above an alien planet, cinematic wide camera movement."
)
seeds=(20261011 20261012 20261013 20261014)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_sensitivity_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --latent-only-names '${LATENTS}' --prompt '${prompts[$index]}' --size 832x448 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 1 --timed-rounds 1 --output-dir '${case_dir}' --master-port '$((30521 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
