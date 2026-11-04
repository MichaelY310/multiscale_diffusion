#!/usr/bin/env bash
# Four-GPU method matrix. Every row keeps model/sampler/seed/video/VSA executor fixed.
set -euo pipefail
ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"
OUT="${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907"

queue_body() {
  local gpu="$1"
  cat <<EOF
set -euo pipefail
cd '${ROOT}'
run() {
  local name="\$1" factors="\$2" mode="\$3" density="\$4" reuse="\$5" scope="\$6"
  if [[ -s '${OUT}/'"\${name}"'.metrics.json' && -s '${OUT}/'"\${name}"'.mp4' ]]; then
    echo "SKIP \${name}"
    return
  fi
  env GPU='${gpu}' NAME="\${name}" FACTORS="\${factors}" ATTENTION_MODE="\${mode}" \
    VSA_SPARSITY='0.8' \
    ROUTE_DENSITY="\${density}" REUSE_EVERY="\${reuse}" REUSE_SCOPE="\${scope}" \
    WARMUP_RUNS='1' TIMED_RUNS='5' bash '${WORKER}'
}
EOF
}

launch_queue() {
  local session="$1" gpu="$2" body="$3"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return 1
  fi
  local script="${OUT}/${session}.sh"
  { queue_body "${gpu}"; printf '%s\n' "${body}"; } > "${script}"
  chmod +x "${script}"
  tmux new-session -d -s "${session}" "bash '${script}' 2>&1 | tee '${OUT}/${session}.log'"
  echo "started ${session} on GPU ${gpu}"
}

launch_queue rdd_fair_matrix_g0 0 "
run fixed_vsa20 '1,1;1,1;1,1' vsa 0.20 1 backend
run fixed_vsa20_reuse2_projected '1,1;1,1;1,1' vsa 0.20 2 projected
run temporal_vsa20 '1,2;1,2;1,1' vsa 0.20 1 backend
run temporal_vsa20_reuse2_projected '1,2;1,2;1,1' vsa 0.20 2 projected
"

launch_queue rdd_fair_matrix_g1 1 "
run temporal_c2f_gate20 '1,2;1,2;1,1' c2f_route 0.20 1 backend
run temporal_c2f_only20 '1,2;1,2;1,1' c2f_route_only 0.20 1 backend
run temporal_c2f_only20_reuse2_projected '1,2;1,2;1,1' c2f_route_only 0.20 2 projected
run temporal_c2f_only12p5_reuse2_projected '1,2;1,2;1,1' c2f_route_only 0.125 2 projected
"

launch_queue rdd_fair_matrix_g2 2 "
run spatial_vsa20 '2,1;2,1;1,1' vsa 0.20 1 backend
run spatial_c2f_gate20 '2,1;2,1;1,1' c2f_route 0.20 1 backend
run spatial_c2f_only20 '2,1;2,1;1,1' c2f_route_only 0.20 1 backend
run spatial_c2f_only20_reuse2_projected '2,1;2,1;1,1' c2f_route_only 0.20 2 projected
"

launch_queue rdd_fair_matrix_g3 3 "
run spatiotemporal_vsa20 '2,2;2,2;1,1' vsa 0.20 1 backend
run spatiotemporal_c2f_gate20 '2,2;2,2;1,1' c2f_route 0.20 1 backend
run spatiotemporal_c2f_only20 '2,2;2,2;1,1' c2f_route_only 0.20 1 backend
run spatiotemporal_c2f_only20_reuse2_projected '2,2;2,2;1,1' c2f_route_only 0.20 2 projected
run spatiotemporal_3stage_c2f_only20 '4,4;2,2;1,1' c2f_route_only 0.20 1 backend
"
