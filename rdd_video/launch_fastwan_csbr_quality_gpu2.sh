#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SESSION="rdd_fastwan_csbr_quality_g2"
OUT="${ROOT}/eval_results/rdd_fastwan_csbr_quality_20260908"
SUITE="${ROOT}/rdd_video/experiments/fastwan_cross_stage_block_reuse_quality.json"
PROMPT="A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."
DECODE="spatial421_rdd_vsa20,csbr_early4_s100,csbr_early6_s075,csbr_early6_s100,csbr_uniform6_s075,csbr_uniform6_s100,csbr_early8_s075,csbr_uniform8_s100,csbr_uniform12_s100"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session ${SESSION} already exists; refusing to alter it." >&2
  exit 1
fi

mkdir -p "${OUT}"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && CUDA_VISIBLE_DEVICES=2 '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name spatial421_rdd_vsa20 --decode-names '${DECODE}' --prompt '${PROMPT}' --size 832x448 --frame-num 29 --warmup-rounds 3 --timed-rounds 12 --output-dir '${OUT}' --master-port 29943 --device cuda:0 2>&1 | tee '${OUT}/run.log'"

echo "Started ${SESSION}; log: ${OUT}/run.log"
