#!/usr/bin/env bash
# Launch one arm of the minimal causal RDD-Wan matrix.
# Usage: bash rdd_video/train_causal_matrix.sh {rdd|boundary|fullres}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARM="${1:-}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/rdd_video_data}"

COMMON_STAGE_CONFIGS="${COMMON_STAGE_CONFIGS:-4,4,20;2,2,20;1,1,10}"
COMMON_STAGE_LENGTHS="${COMMON_STAGE_LENGTHS:-60,350,590}"

case "${ARM}" in
  rdd)
    EXP_NAME="${EXP_NAME:-video_p0_rdd_drift1}"
    DATA_DIR="${DATA_DIR:-${DATA_ROOT}/multires_latents_latent}"
    DATA_MODE="multires"
    STAGE_CONFIGS="${COMMON_STAGE_CONFIGS}"
    STAGE_LENGTHS="${COMMON_STAGE_LENGTHS}"
    DRIFT_SCALE="1.0"
    ;;
  boundary)
    # Same data, resolutions, stage sampling, optimizer and updates as RDD;
    # only the residualized interior path is removed.
    EXP_NAME="${EXP_NAME:-video_p0_boundary_drift0}"
    DATA_DIR="${DATA_DIR:-${DATA_ROOT}/multires_latents_latent}"
    DATA_MODE="multires"
    STAGE_CONFIGS="${COMMON_STAGE_CONFIGS}"
    STAGE_LENGTHS="${COMMON_STAGE_LENGTHS}"
    DRIFT_SCALE="0.0"
    ;;
  fullres)
    # Same terminal teacher latents and update count, but standard full-res FM.
    # Report GPU-hours separately; this arm is intentionally more expensive.
    EXP_NAME="${EXP_NAME:-video_p0_fullres_finetune}"
    DATA_DIR="${DATA_DIR:-${DATA_ROOT}/synthetic_wan1p3b/latent}"
    DATA_MODE="single"
    STAGE_CONFIGS="1,1,50"
    STAGE_LENGTHS="1000"
    DRIFT_SCALE="0.0"
    ;;
  *)
    echo "Usage: bash ${BASH_SOURCE[0]} {rdd|boundary|fullres}" >&2
    exit 2
    ;;
esac

echo "Launching causal arm: ${ARM} (${EXP_NAME})"
env \
  EXP_NAME="${EXP_NAME}" \
  DATA_DIR="${DATA_DIR}" \
  DATA_MODE="${DATA_MODE}" \
  STAGE_CONFIGS="${STAGE_CONFIGS}" \
  STAGE_LENGTHS="${STAGE_LENGTHS}" \
  DRIFT_SCALE="${DRIFT_SCALE}" \
  NUM_GPUS="${NUM_GPUS:-2}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-12}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
  STAGE_SAMPLING="${STAGE_SAMPLING:-balanced}" \
  MAX_STEPS="${MAX_STEPS:-500}" \
  WARMUP_STEPS="${WARMUP_STEPS:-0}" \
  LR="${LR:-1e-5}" \
  SEED="${SEED:-42}" \
  bash "${SCRIPT_DIR}/train_rdd_wan.sh"
