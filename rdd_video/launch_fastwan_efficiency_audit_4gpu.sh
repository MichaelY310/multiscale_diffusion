#!/usr/bin/env bash
# Stable, paired FastWan/RDD timing audit.  Each process loads the model once,
# warms every candidate, alternates candidate order, and reports medians plus
# all raw runs.  This script owns only the four tmux sessions named below.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_fastwan_efficiency_audit_20260908}"
PROMPT="A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."
mkdir -p "${OUT}"

launch() {
  local session="$1" gpu="$2" suite="$3" size="$4" warmup="$5" rounds="$6" decode="$7" port="$8"
  local run_dir="${OUT}/${session}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to replace existing session: ${session}" >&2
    return 1
  fi
  mkdir -p "${run_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${suite}' --baseline-name fixed_vsa20 --prompt '${PROMPT}' --size '${size}' --frame-num 29 --warmup-rounds '${warmup}' --timed-rounds '${rounds}' --decode-names '${decode}' --output-dir '${run_dir}' --master-port '${port}' --device cuda:0 2>&1 | tee '${run_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}: ${size}, ${rounds} timed rounds"
}

SPATIAL="rdd_video/experiments/fastwan_spatial_efficiency_suite.json"
SPATIOTEMPORAL="rdd_video/experiments/fastwan_spatiotemporal_efficiency_suite.json"
SPATIAL_DECODE="fixed_vsa20,spatial221_c2f20,spatial421_c2f20,spatial421_c2f18p75,spatial441_c2f20,spatial441_c2f20_reuse2,spatial441_c2f12p5_reuse2,spatial421_c2f20_dense_seed,spatial421_c2f20_reuse_coarse"
ST_DECODE="fixed_vsa20,st221_c2f20,st421_c2f20,st421_c2f12p5,st441_c2f20,st441_c2f20_reuse2,st441_c2f12p5_reuse2,st421_c2f20_dense_seed,st421_c2f20_reuse_coarse"

launch rdd_eff_spatial832_g0 0 "${SPATIAL}" 832x448 5 50 "${SPATIAL_DECODE}" 29810
launch rdd_eff_st832_g1 1 "${SPATIOTEMPORAL}" 832x448 5 50 "${ST_DECODE}" 29811
launch rdd_eff_spatial1280_g2 2 "${SPATIAL}" 1280x704 3 30 "" 29812
launch rdd_eff_spatial1664_g3 3 "${SPATIAL}" 1664x896 3 30 "" 29813

