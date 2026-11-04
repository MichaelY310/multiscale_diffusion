#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
WORKER="$ROOT/rdd_video/run_fastwan_fair_worker.sh"
GPU="${GPU:?set physical GPU id (0-3)}"
SUITE="${SUITE:?set SUITE=city or coast}"

case "$SUITE" in
  city)
    PROMPT="A young woman wearing a bright yellow raincoat walks through a busy neon-lit street at night, reflections on wet pavement, smooth tracking shot."
    SEED=20260911
    ;;
  coast)
    PROMPT="A vintage blue car drives along a winding coastal road at sunset, ocean waves and cliffs in the background, cinematic aerial camera movement."
    SEED=20260912
    ;;
  *)
    echo "Unknown SUITE=$SUITE (expected city or coast)" >&2
    exit 2
    ;;
esac

run_one() {
  local suffix="$1"
  local factors="$2"
  GPU="$GPU" NAME="quality_${SUITE}_${suffix}" FACTORS="$factors" \
    ATTENTION_MODE=c2f_route_only VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 \
    REUSE_EVERY=1 REUSE_SCOPE=backend WARMUP_RUNS=1 TIMED_RUNS=5 \
    SEED="$SEED" PROMPT="$PROMPT" bash "$WORKER"
}

# These are compact visual-transfer runs.  Formal latency comes from the
# paired fox timing jobs, not from cross-GPU comparison of this queue.
run_one temporal421_c2f20        "1,4;1,2;1,1"
run_one spatial221_c2f20         "2,1;2,1;1,1"
run_one spatial421_c2f20         "4,1;2,1;1,1"
run_one spatiotemporal221_c2f20  "2,2;2,2;1,1"
run_one spatiotemporal421_c2f20  "4,4;2,2;1,1"

