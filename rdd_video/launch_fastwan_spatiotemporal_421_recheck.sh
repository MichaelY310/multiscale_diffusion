#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="$ROOT/rdd_video/run_fastwan_fair_worker.sh"
GPU="${GPU:-2}"

run_one() {
  local name="$1"
  local factors="$2"
  local mode="$3"
  GPU="$GPU" NAME="$name" FACTORS="$factors" ATTENTION_MODE="$mode" \
    VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 REUSE_EVERY=1 REUSE_SCOPE=backend \
    WARMUP_RUNS=5 TIMED_RUNS=20 SEED=20260905 \
    bash "$WORKER"
}

# A same-GPU pre/method/post sandwich removes clock and co-tenant drift from
# the comparison.  Nothing in this script searches for or signals processes.
run_one grid_spatiotemporal_421_c2f20_recheck_fixed_pre  "1,1;1,1;1,1" vsa
run_one grid_spatiotemporal_421_c2f20_recheck            "4,4;2,2;1,1" c2f_route_only
run_one grid_spatiotemporal_421_c2f20_recheck_fixed_post "1,1;1,1;1,1" vsa

