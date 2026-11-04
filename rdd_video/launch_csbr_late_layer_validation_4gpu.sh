#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_late_layer_candidates.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_late_layer_validation_20260910"
DECODE="rdd_vsa20_baseline,uniform6_global075,safe4_multiscale,safe6_multiscale,safe8_multiscale"
LATENTS="rdd_vsa20_baseline,uniform6_global075,safe4_multiscale,safe6_scalar,safe6_multiscale,tail6_multiscale,safe8_multiscale"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A glassblower shapes glowing molten glass inside a dark workshop, sparks drifting through the air, slow cinematic camera movement."
  "A herd of horses runs across a green valley beneath dramatic clouds, smooth aerial tracking shot."
  "A tiny orange submarine explores a coral reef filled with colorful fish, gentle underwater camera movement."
  "A dancer in a flowing blue dress performs on a rooftop at sunset while fabric moves in the wind, cinematic shot."
)
seeds=(20261021 20261022 20261023 20261024)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_late_validation_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --decode-names '${DECODE}' --latent-only-names '${LATENTS}' --prompt '${prompts[$index]}' --size 832x448 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 3 --timed-rounds 7 --output-dir '${case_dir}' --master-port '$((30541 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
