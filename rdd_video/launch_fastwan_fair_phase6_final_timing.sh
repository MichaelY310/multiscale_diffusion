#!/usr/bin/env bash
# Final single-GPU paired timing: baseline -> ablations -> baseline.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"
OUT="${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907"
SESSION="rdd_fair_kernel_final_timing"

run() {
  local name="$1" factors="$2" mode="$3" reuse="$4" scope="$5"
  env GPU=0 NAME="${name}" FACTORS="${factors}" ATTENTION_MODE="${mode}" \
    VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 REUSE_EVERY="${reuse}" \
    REUSE_SCOPE="${scope}" WARMUP_RUNS=3 TIMED_RUNS=20 bash "${WORKER}"
}

if [[ "${RDD_TIMING_CHILD:-0}" == "1" ]]; then
  run final_fixed_vsa20_pre '1,1;1,1;1,1' vsa 1 backend
  run final_temporal_vsa20 '1,2;1,2;1,1' vsa 1 backend
  run final_temporal_c2f20 '1,2;1,2;1,1' c2f_route_only 1 backend
  run final_temporal_c2f20_reuse2 '1,2;1,2;1,1' c2f_route_only 2 projected
  run final_fixed_vsa20_post '1,1;1,1;1,1' vsa 1 backend
  exit 0
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "session already exists: ${SESSION}" >&2
  exit 1
fi
tmux new-session -d -s "${SESSION}" \
  "env RDD_TIMING_CHILD=1 bash '${ROOT}/rdd_video/launch_fastwan_fair_phase6_final_timing.sh'"
echo "started ${SESSION} on GPU 0"
