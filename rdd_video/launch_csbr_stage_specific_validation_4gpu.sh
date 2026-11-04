#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SUITE="${ROOT}/rdd_video/experiments/fastwan_csbr_stage_specific_validation.json"
OUT_ROOT="${ROOT}/eval_results/rdd_csbr_stage_specific_validation_20260910"
DECODE="rdd_vsa20_baseline,middle_safe2,fine_safe6,combined_disjoint_m2_f6,combined_overlap_m2_f6,combined_disjoint_m4_f4"
LATENTS="rdd_vsa20_baseline,middle_safe2,middle_safe4,fine_safe6,fine_disjoint6,combined_disjoint_m2_f6,combined_overlap_m2_f6,combined_disjoint_m4_f4"

cd "${ROOT}"
"${PYTHON}" rdd_video/build_csbr_stage_specific_validation_configs.py --output "${SUITE}"
mkdir -p "${OUT_ROOT}"

prompts=(
  "A polar bear walks across cracked sea ice beneath the northern lights, slow cinematic camera movement."
  "A vintage tram moves through a rainy European street at dusk, reflections glowing on wet pavement."
  "A hummingbird hovers beside red flowers in a sunlit garden, smooth macro tracking shot."
  "A climber reaches a mountain summit above a sea of clouds at sunrise, wide aerial shot."
)
seeds=(20261121 20261122 20261123 20261124)

for index in 0 1 2 3; do
  gpu="${index}"
  session="rdd_csbr_stage_specific_g${gpu}"
  case_dir="${OUT_ROOT}/case_${index}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "ERROR: tmux session already exists: ${session}" >&2
    exit 1
  fi
  mkdir -p "${case_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name rdd_vsa20_baseline --decode-names '${DECODE}' --latent-only-names '${LATENTS}' --prompt '${prompts[$index]}' --size 832x448 --frame-num 29 --seed '${seeds[$index]}' --warmup-rounds 4 --timed-rounds 12 --output-dir '${case_dir}' --master-port '$((30641 + index))' --device cuda:0 2>&1 | tee '${case_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${case_dir}/run.log"
done
