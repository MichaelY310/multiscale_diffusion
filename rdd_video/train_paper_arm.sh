#!/usr/bin/env bash
# Train one paper-scale causal arm with the PyramidalWan compute/data budget.
# Usage: bash rdd_video/train_paper_arm.sh {rdd|boundary|fullres}

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ARM="${1:-}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/rdd_video_data_paper}"

case "${ARM}" in
  rdd)
    EXP_NAME="${EXP_NAME:-video_paper_rdd_drift1}"
    DATA_DIR="${DATA_DIR:-${DATA_ROOT}/multires_latents_latent_wan14b_80k}"
    DATA_MODE=multires
    STAGE_CONFIGS="4,4,20;2,2,20;1,1,10"
    STAGE_LENGTHS="60,350,590"
    DRIFT_SCALE=1.0
    ;;
  boundary)
    EXP_NAME="${EXP_NAME:-video_paper_boundary_drift0}"
    DATA_DIR="${DATA_DIR:-${DATA_ROOT}/multires_latents_latent_wan14b_80k}"
    DATA_MODE=multires
    STAGE_CONFIGS="4,4,20;2,2,20;1,1,10"
    STAGE_LENGTHS="60,350,590"
    DRIFT_SCALE=0.0
    ;;
  fullres)
    EXP_NAME="${EXP_NAME:-video_paper_fullres_finetune}"
    DATA_DIR="${DATA_DIR:-${DATA_ROOT}/synthetic_wan14b_80k/latent}"
    DATA_MODE=single
    STAGE_CONFIGS="1,1,50"
    STAGE_LENGTHS=1000
    DRIFT_SCALE=0.0
    ;;
  *)
    echo "Usage: bash ${BASH_SOURCE[0]} {rdd|boundary|fullres}" >&2
    exit 2
    ;;
esac

# PyramidalWan uses 2 H100s, batch 6/GPU and 5K iterations. Their mixed batch
# contains 2 examples/stage/GPU. Sampling one whole stage uniformly per update
# gives the same expected 20K examples per stage over 5K updates:
# 5000 * (1/3) * global_batch_12 = 20000.
env \
  PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}" \
  DATA_DIR="${DATA_DIR}" \
  DATA_MODE="${DATA_MODE}" \
  EXP_NAME="${EXP_NAME}" \
  RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results_rdd_video_paper}" \
  NUM_GPUS="${NUM_GPUS:-2}" \
  GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-12}" \
  MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}" \
  MAX_STEPS="${MAX_STEPS:-5000}" \
  WARMUP_STEPS="${WARMUP_STEPS:-0}" \
  LR="${LR:-1e-5}" \
  STAGE_CONFIGS="${STAGE_CONFIGS}" \
  STAGE_LENGTHS="${STAGE_LENGTHS}" \
  STAGE_SAMPLING="${STAGE_SAMPLING:-balanced}" \
  DRIFT_SCALE="${DRIFT_SCALE}" \
  SEED="${SEED:-42}" \
  bash "${SCRIPT_DIR}/train_rdd_wan.sh"
