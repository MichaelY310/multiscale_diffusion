#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "usage: $0 GPU TAG WIDTHxHEIGHT PROMPT" >&2
  exit 2
fi

GPU="$1"
TAG="$2"
SIZE="$3"
PROMPT="$4"
ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
SESSION="rdd_csbr_${TAG}_g${GPU}"
OUT="${ROOT}/eval_results/rdd_csbr_verify_${TAG}"
SUITE="${SUITE_OVERRIDE:-${ROOT}/rdd_video/experiments/fastwan_cross_stage_block_reuse_verify.json}"
DECODE="${DECODE_OVERRIDE:-spatial421_rdd_vsa20,csbr_uniform6_s075,csbr_uniform12_s100}"
WARMUP="${WARMUP:-5}"
TIMED="${TIMED:-20}"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "ERROR: tmux session ${SESSION} already exists; refusing to alter it." >&2
  exit 1
fi

mkdir -p "${OUT}"
tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${GPU}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name spatial421_rdd_vsa20 --prompt '${PROMPT}' --size '${SIZE}' --frame-num 29 --warmup-rounds '${WARMUP}' --timed-rounds '${TIMED}' --decode-names '${DECODE}' --output-dir '${OUT}' --master-port '$((30000 + GPU))' --device cuda:0 2>&1 | tee '${OUT}/run.log'"

echo "Started ${SESSION}; log: ${OUT}/run.log"
