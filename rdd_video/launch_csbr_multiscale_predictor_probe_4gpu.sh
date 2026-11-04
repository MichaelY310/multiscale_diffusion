#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_multiscale_predictor_probe_20260909"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."
  "A small robot carefully assembles a wooden toy on a workbench, warm workshop lighting, close tracking shot."
  "A white sailboat crosses turquoise ocean waves beside steep green cliffs, smooth aerial camera movement."
  "An astronaut in a white suit walks across a red desert under two moons, cinematic wide shot."
)
seeds=(20260931 20260932 20260933 20260934)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_predictor_probe_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/diagnose_csbr_residual_hypotheses.py --root '${ROOT}' --prompt '${prompts[$index]}' --seed '${seeds[$index]}' --size 832x448 --frame-num 29 --master-port '$((30401 + index))' --probe-only --output-dir '${case_dir}' 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done

