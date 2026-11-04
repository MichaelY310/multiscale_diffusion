#!/usr/bin/env bash
# Small multi-prompt quality sanity matrix for the four main fair methods.
# Timing claims come from phase 6; this phase is for visible failure patterns.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="${ROOT}/rdd_video/run_fastwan_fair_worker.sh"

run_case() {
  local gpu="$1" method="$2" factors="$3" mode="$4" reuse="$5" scope="$6"
  local prompt1='A young woman wearing a bright yellow raincoat walks through a busy neon-lit street at night, reflections on wet pavement, smooth tracking shot.'
  local prompt2='A vintage blue car drives along a winding coastal road at sunset, ocean waves and cliffs in the background, cinematic aerial camera movement.'
  env GPU="${gpu}" NAME="quality_city_${method}" FACTORS="${factors}" \
    ATTENTION_MODE="${mode}" VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 \
    REUSE_EVERY="${reuse}" REUSE_SCOPE="${scope}" WARMUP_RUNS=1 TIMED_RUNS=3 \
    SEED=20260911 PROMPT="${prompt1}" bash "${WORKER}"
  env GPU="${gpu}" NAME="quality_coast_${method}" FACTORS="${factors}" \
    ATTENTION_MODE="${mode}" VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 \
    REUSE_EVERY="${reuse}" REUSE_SCOPE="${scope}" WARMUP_RUNS=1 TIMED_RUNS=3 \
    SEED=20260912 PROMPT="${prompt2}" bash "${WORKER}"
}

if [[ "${RDD_QUALITY_CHILD:-}" != "" ]]; then
  case "${RDD_QUALITY_CHILD}" in
    0) run_case 0 fixed_vsa20 '1,1;1,1;1,1' vsa 1 backend ;;
    1) run_case 1 temporal_vsa20 '1,2;1,2;1,1' vsa 1 backend ;;
    2) run_case 2 temporal_c2f20 '1,2;1,2;1,1' c2f_route_only 1 backend ;;
    3) run_case 3 temporal_c2f20_reuse2 '1,2;1,2;1,1' c2f_route_only 2 projected ;;
    *) echo "unknown RDD_QUALITY_CHILD=${RDD_QUALITY_CHILD}" >&2; exit 2 ;;
  esac
  exit 0
fi

for gpu in 0 1 2 3; do
  session="rdd_fair_quality_g${gpu}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "session already exists: ${session}" >&2
    exit 1
  fi
done

for gpu in 0 1 2 3; do
  session="rdd_fair_quality_g${gpu}"
  tmux new-session -d -s "${session}" \
    "env RDD_QUALITY_CHILD='${gpu}' bash '${ROOT}/rdd_video/launch_fastwan_fair_phase7_quality.sh'"
  echo "started ${session} on GPU ${gpu}"
done
