#!/usr/bin/env bash
# Native-ImageNet Unified DiT-L/2 two-stage RDD ablation on GPUs 0-3.
#
# Forward-process order (finest -> coarsest): 32x32,16x16 with 500/500
# diffusion timesteps. Sampling order is therefore 16x16 -> 32x32.
# All other training settings match the Unified-L boundary runs:
# global batch 256, uniform timestep sampling, unweighted RDD loss, BF16,
# original ImageNet, EMA VAE, seed 0, and 400K optimizer updates.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"

export RDD_UNIFIED_STAGE_LENGTHS="500,500"
export RDD_UNIFIED_RESOLUTIONS="32,16"
export RDD_UNIFIED_RUN_NAME="RDD-Unified-L-2Stage-500-500"
export RDD_UNIFIED_DEFAULT_GPUS="0,1,2,3"
export RDD_UNIFIED_DEFAULT_MASTER_PORT="29651"
export RDD_UNIFIED_LOG_STEM="train_rdd_unified_l_2stage_500_500_4gpu_la9"

# The shared launcher defaults to 07:00. This experiment instead follows the
# DMD resource window and stops at the next 09:00 America/Los_Angeles. An
# explicit caller-provided STOP_AT_EPOCH still takes precedence.
if [[ -z "${STOP_AT_EPOCH:-}" ]]; then
    STOP_AT_EPOCH="$(python3 - <<'PY'
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

zone = ZoneInfo("America/Los_Angeles")
now = datetime.now(zone)
target = now.replace(hour=9, minute=0, second=0, microsecond=0)
if target <= now:
    target += timedelta(days=1)
print(int(target.timestamp()))
PY
)"
    export STOP_AT_EPOCH
fi

exec bash "${REPO_ROOT}/2d/train_RDD_unified_L_boundary_4gpu_sf7.sh"
