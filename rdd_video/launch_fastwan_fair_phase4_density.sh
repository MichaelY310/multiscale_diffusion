#!/usr/bin/env bash
# Matched-density, paired-baseline sweep.  Each GPU runs one density from a
# VSA baseline through RDD/C2F/reuse variants and repeats the VSA baseline.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"
OUT="${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907"

queue_body() {
  local gpu="$1" density="$2" sparsity="$3" tag="$4"
  cat <<EOF
set -euo pipefail
cd '${ROOT}'
run() {
  local name="\$1" factors="\$2" mode="\$3" reuse="\$4" scope="\$5"
  env GPU='${gpu}' NAME="\${name}" FACTORS="\${factors}" ATTENTION_MODE="\${mode}" \
    VSA_SPARSITY='${sparsity}' ROUTE_DENSITY='${density}' \
    REUSE_EVERY="\${reuse}" REUSE_SCOPE="\${scope}" \
    WARMUP_RUNS='2' TIMED_RUNS='10' bash '${WORKER}'
}
run pair_d${tag}_fixed_vsa_pre '1,1;1,1;1,1' vsa 1 backend
run pair_d${tag}_temporal_vsa '1,2;1,2;1,1' vsa 1 backend
run pair_d${tag}_temporal_c2f_only '1,2;1,2;1,1' c2f_route_only 1 backend
run pair_d${tag}_temporal_c2f_reuse2 '1,2;1,2;1,1' c2f_route_only 2 projected
run pair_d${tag}_fixed_vsa_post '1,1;1,1;1,1' vsa 1 backend
EOF
}

launch_queue() {
  local session="$1" gpu="$2" density="$3" sparsity="$4" tag="$5"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return 1
  fi
  local script="${OUT}/${session}.sh"
  queue_body "${gpu}" "${density}" "${sparsity}" "${tag}" > "${script}"
  chmod +x "${script}"
  tmux new-session -d -s "${session}" \
    "bash '${script}' 2>&1 | tee '${OUT}/${session}.log'"
  echo "started ${session} on GPU ${gpu}: keep=${density}"
}

launch_queue rdd_fair_density_g0 0 0.50 0.50 50
launch_queue rdd_fair_density_g1 1 0.25 0.75 25
launch_queue rdd_fair_density_g2 2 0.20 0.80 20
launch_queue rdd_fair_density_g3 3 0.125 0.875 12p5

