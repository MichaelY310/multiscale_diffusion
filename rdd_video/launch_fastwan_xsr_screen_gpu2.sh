#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SESSION="rdd_fastwan_xsr_screen_g2"
OUT="${ROOT}/eval_results/rdd_fastwan_xsr_screen_20260909"
SUITE="${ROOT}/rdd_video/experiments/fastwan_cross_stage_branch_reuse_screen.json"
PROMPT="A red fox walks through a snowy forest, cinematic camera movement."
DECODE="spatial421_rdd_vsa20,xsr_self_u12_s075,xsr_mlp_u12_s075,xsr_self_mlp_u6_s075,xsr_self_mlp_u8_s075"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session ${SESSION} already exists" >&2
  exit 1
fi

mkdir -p "${OUT}"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && CUDA_VISIBLE_DEVICES=2 '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name spatial421_rdd_vsa20 --prompt '${PROMPT}' --size 832x448 --frame-num 29 --warmup-rounds 3 --timed-rounds 12 --decode-names '${DECODE}' --output-dir '${OUT}' --master-port 30102 --device cuda:0 2>&1 | tee '${OUT}/run.log'"

echo "Started ${SESSION}; log: ${OUT}/run.log"
