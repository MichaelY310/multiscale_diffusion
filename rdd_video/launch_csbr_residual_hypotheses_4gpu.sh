#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: missing Python: ${PYTHON}" >&2
  exit 1
fi

OUT_ROOT="${ROOT}/eval_results/rdd_csbr_residual_hypotheses_20260909"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A young woman wearing a bright yellow raincoat walks through a busy neon-lit street at night, reflections on wet pavement, smooth tracking shot."
  "A vintage blue car drives along a winding coastal road at sunset, ocean waves and cliffs in the background, cinematic aerial camera movement."
  "A young woman wearing a bright yellow raincoat walks through a busy neon-lit street at night, reflections on wet pavement, smooth tracking shot."
  "A vintage blue car drives along a winding coastal road at sunset, ocean waves and cliffs in the background, cinematic aerial camera movement."
)
seeds=(20260921 20260922 20260923 20260924)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_residual_diag_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  decode=""
  if [[ "${index}" == "0" ]]; then
    decode="--decode"
  fi
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  cmd="cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/diagnose_csbr_residual_hypotheses.py --root '${ROOT}' --prompt '${prompts[$index]}' --seed '${seeds[$index]}' --size 832x448 --frame-num 29 --master-port '$((30341 + index))' --output-dir '${case_dir}' ${decode} 2>&1 | tee '${case_dir}/run.log'"
  tmux new-session -d -s "${session}" "${cmd}"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
