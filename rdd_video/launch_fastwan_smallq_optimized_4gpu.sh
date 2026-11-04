#!/usr/bin/env bash
# End-to-end validation of the exact small-query Triton sparse executor.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_fastwan_smallq_optimized_20260908}"
SUITE="rdd_video/experiments/fastwan_smallq_optimized_suite.json"
FOX="A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."
CITY="A person in a yellow raincoat walks through a neon-lit city street at night, wet pavement reflections, cinematic camera movement."
COAST="A blue vintage car drives along a coastal road at sunset, ocean waves and cliffs in the background, cinematic tracking shot."
DECODE="fixed_vsa20,spatial221_c2f20,spatial421_c2f20,spatial421_c2f20_q32,spatial441_c2f20,spatial441_c2f20_q32_reuse2,spatial441_c2f12p5_q32_reuse2,st221_c2f20,st421_c2f20,st421_c2f20_q32,st441_c2f20_q32_reuse2,st441_c2f12p5_q32_reuse2"
mkdir -p "${OUT}"

launch() {
  local session="$1" gpu="$2" prompt="$3" size="$4" warmup="$5" rounds="$6" decode="$7" port="$8"
  local run_dir="${OUT}/${session}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to replace existing session: ${session}" >&2
    return 1
  fi
  mkdir -p "${run_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name fixed_vsa20 --prompt '${prompt}' --size '${size}' --frame-num 29 --warmup-rounds '${warmup}' --timed-rounds '${rounds}' --decode-names '${decode}' --output-dir '${run_dir}' --master-port '${port}' --device cuda:0 2>&1 | tee '${run_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}"
}

launch rdd_smallq_fox832_g0 0 "${FOX}" 832x448 5 50 "${DECODE}" 29830
launch rdd_smallq_large1280_g1 1 "${FOX}" 1280x704 3 30 "" 29831
launch rdd_smallq_large1664_g2 2 "${FOX}" 1664x896 3 30 "" 29832
launch rdd_smallq_city832_g3 3 "${CITY}" 832x448 3 20 "${DECODE}" 29833

