#!/usr/bin/env bash
# Add the two stage-factor paths missing from the fair FastWan sweep.
#
# The three other requested paths already exist in the same result directory:
#   (2,1)->(2,1)->(1,1)  spatial_c2f_only20
#   (2,2)->(2,2)->(1,1)  spatiotemporal_c2f_only20
#   (4,4)->(2,2)->(1,1)  spatiotemporal_3stage_c2f_only20_triton
#
# Each new path is bracketed by the identical fixed-resolution VSA baseline on
# the same physical GPU. This makes the relative speed usable even when GPU
# clocks or background load differ between cards.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
OUT="${ROOT}/eval_results/rdd_fastwan_fair_accel_20260907"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"

launch_pair() {
  local session="$1" gpu="$2" stem="$3" factors="$4"
  local payload="${OUT}/${session}.sh"
  local log="${OUT}/${session}.log"

  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    return 1
  fi

  mkdir -p "${OUT}"
  {
    echo '#!/usr/bin/env bash'
    echo 'set -euo pipefail'
    printf 'cd %q\n' "${ROOT}"
    for suffix in fixed_pre method fixed_post; do
      local name mode run_factors
      if [[ "${suffix}" == method ]]; then
        name="${stem}"
        mode="c2f_route_only"
        run_factors="${factors}"
      else
        name="${stem}_${suffix}"
        mode="vsa"
        run_factors='1,1;1,1;1,1'
      fi
      printf "env GPU=%q NAME=%q FACTORS=%q ATTENTION_MODE=%q VSA_SPARSITY=0.8 ROUTE_DENSITY=0.20 REUSE_EVERY=1 REUSE_SCOPE=backend WARMUP_RUNS=5 TIMED_RUNS=20 bash %q\n" \
        "${gpu}" "${name}" "${run_factors}" "${mode}" "${WORKER}"
    done
  } > "${payload}"
  chmod +x "${payload}"
  tmux new-session -d -s "${session}" "bash '${payload}' 2>&1 | tee '${log}'"
  echo "started ${session} on physical GPU ${gpu}; log=${log}"
}

launch_pair \
  rdd_fastwan_grid_temporal421 \
  "${GPU_TEMPORAL:-0}" \
  grid_temporal_4x2x1_c2f20 \
  '1,4;1,2;1,1'

launch_pair \
  rdd_fastwan_grid_spatial421 \
  "${GPU_SPATIAL:-1}" \
  grid_spatial_4x2x1_c2f20 \
  '4,1;2,1;1,1'
