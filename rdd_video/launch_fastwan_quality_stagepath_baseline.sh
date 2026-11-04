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
  *) exit 2 ;;
esac

GPU="$GPU" NAME="quality_${SUITE}_stagepaths_fixed_post" \
  FACTORS="1,1;1,1;1,1" ATTENTION_MODE=vsa \
  VSA_SPARSITY=0.8 ROUTE_DENSITY=0.2 REUSE_EVERY=1 REUSE_SCOPE=backend \
  WARMUP_RUNS=2 TIMED_RUNS=10 SEED="$SEED" PROMPT="$PROMPT" \
  bash "$WORKER"

