#!/usr/bin/env bash
# Evaluate the completed Unified-L direct 8x8 -> 32x32, 500/500 run.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT_DIR="${REPO_ROOT}/checkpoints/RDD-Unified-L-2Stage-8to32-500-500/imagenet256-unified/checkpoints"
export EVAL_ROOT="${REPO_ROOT}/eval_results/unified_l_2stage_8to32_500500_convergence"
export RESOLUTIONS="32,8"
export RUN_LABEL="8to32_500500"
export TAG_PREFIX="unifiedL_8to32_500500"
export GPU_IDS="${GPU_IDS:-0,1}"
export WORLD_SIZE="${WORLD_SIZE:-2}"

exec bash "${SCRIPT_DIR}/eval_unified_l_2stage_500500_convergence_50k_400k_4gpu.sh"
