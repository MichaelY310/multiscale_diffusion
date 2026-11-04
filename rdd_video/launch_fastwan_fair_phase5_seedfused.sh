#!/usr/bin/env bash
# Regression/timing run for the single-pass VSA cascade seeding optimization.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT="${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907"
SESSION="rdd_fair_kernel_seedfused"

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 1
fi

tmux new-session -d -s "${SESSION}" \
  "cd '${ROOT}' && env GPU=0 NAME=temporal_c2f_only20_reuse2_projected_seedfused FACTORS='1,2;1,2;1,1' ATTENTION_MODE=c2f_route_only VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 REUSE_EVERY=2 REUSE_SCOPE=projected WARMUP_RUNS=3 TIMED_RUNS=20 bash rdd_video/run_fastwan_fair_worker.sh"

echo "started ${SESSION} on GPU 0"
