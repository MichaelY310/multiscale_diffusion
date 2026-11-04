#!/usr/bin/env bash
# Validate the fused Triton C2F route lift on the strongest phase-2 settings.
# Each worker uses one physical GPU and otherwise keeps the FastWan setup fixed.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"
OUT="${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907"

launch() {
  local session="$1" gpu="$2" name="$3" factors="$4" mode="$5" density="$6" reuse="$7" scope="$8"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return 1
  fi
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && env GPU='${gpu}' NAME='${name}' FACTORS='${factors}' ATTENTION_MODE='${mode}' VSA_SPARSITY='0.8' ROUTE_DENSITY='${density}' REUSE_EVERY='${reuse}' REUSE_SCOPE='${scope}' WARMUP_RUNS='2' TIMED_RUNS='10' bash '${WORKER}' 2>&1 | tee '${OUT}/${session}.log'"
  echo "started ${session} on GPU ${gpu}"
}

launch rdd_fair_triton_g0 0 temporal_c2f_only20_reuse2_projected_triton \
  '1,2;1,2;1,1' c2f_route_only 0.20 2 projected
launch rdd_fair_triton_g1 1 spatial_c2f_only20_reuse2_projected_triton \
  '2,1;2,1;1,1' c2f_route_only 0.20 2 projected
launch rdd_fair_triton_g2 2 spatiotemporal_c2f_only20_reuse2_projected_triton \
  '2,2;2,2;1,1' c2f_route_only 0.20 2 projected
launch rdd_fair_triton_g3 3 spatiotemporal_3stage_c2f_only20_triton \
  '4,4;2,2;1,1' c2f_route_only 0.20 1 backend

