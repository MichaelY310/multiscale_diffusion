#!/usr/bin/env bash
# Isolated, paired resident timing for online VSA-20 versus C2F near 20%.
# This launcher owns only the tmux session named below and never signals it.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SESSION="rdd_c2f20_resident_recheck"
OUT="${ROOT}/eval_results/rdd_c2f20_resident_recheck_20260908"
CONFIGS="${ROOT}/rdd_video/experiments/fastwan_resident_c2f_20pct_isolation_suite.json"
PROMPT="A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "Refusing to replace existing session: ${SESSION}" >&2
  exit 1
fi

mkdir -p "${OUT}"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && CUDA_VISIBLE_DEVICES=0 '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${CONFIGS}' --baseline-name spatial421_online_vsa20 --prompt '${PROMPT}' --size 832x448 --frame-num 29 --warmup-rounds 10 --timed-rounds 100 --output-dir '${OUT}' --master-port 29931 --device cuda:0 2>&1 | tee '${OUT}/run.log'"

echo "Started ${SESSION}; result: ${OUT}/paired_benchmark.json"
