#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_layerwise_predictor.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_layerwise_predictor_20260910"
DECODE="rdd_vsa20_baseline,uniform6_global075,top6_layer_scalar,top6_multiscale,top8_multiscale_high"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."
  "A small robot carefully assembles a wooden toy on a workbench, warm workshop lighting, close tracking shot."
  "A white sailboat crosses turquoise ocean waves beside steep green cliffs, smooth aerial camera movement."
  "An astronaut in a white suit walks across a red desert under two moons, cinematic wide shot."
)
seeds=(20261001 20261002 20261003 20261004)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_layerwise_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --decode-names '${DECODE}' --prompt '${prompts[$index]}' --size 832x448 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 2 --timed-rounds 5 --output-dir '${case_dir}' --master-port '$((30501 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
