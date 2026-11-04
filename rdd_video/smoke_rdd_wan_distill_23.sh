#!/usr/bin/env bash
# Two-GPU plumbing test. It may use base Wan only because it does not claim quality.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_ENV_DIR="${VIDEO_ENV_DIR:-${REPO_DIR}/.venv-rdd-video}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results_rdd_video_distill_smoke}"
EXP_NAME="${EXP_NAME:-gpu23_contract}"

source "${VIDEO_ENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONPATH="${REPO_DIR}/Wan2.1:${REPO_DIR}/rdd_video:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

mkdir -p "${RESULTS_DIR}/${EXP_NAME}"
torchrun --standalone --nproc_per_node=2 "${SCRIPT_DIR}/distill_rdd_wan.py" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --allow-base-teacher \
  --data-mode dummy \
  --stage-configs "4,4,2;2,2,2;1,1,1" \
  --stage-lengths "60,350,590" \
  --student-steps "2,2,1" \
  --trajectory-steps "${TRAJECTORY_STEPS:-1}" \
  --max-steps "${MAX_STEPS:-1}" \
  --teacher-substeps "${TEACHER_SUBSTEPS:-1}" \
  --teacher-solver euler \
  --teacher-cfg-scale 1 \
  --lora-rank "${LORA_RANK:-4}" \
  --lora-alpha "${LORA_ALPHA:-4}" \
  --critic-lora-rank "${CRITIC_LORA_RANK:-4}" \
  --critic-lora-alpha "${CRITIC_LORA_ALPHA:-4}" \
  --micro-batch-size 1 \
  --grad-accum-steps 1 \
  --num-workers 0 \
  --smoke-latent-frames "${SMOKE_LATENT_FRAMES:-9}" \
  --smoke-latent-height "${SMOKE_LATENT_HEIGHT:-16}" \
  --smoke-latent-width "${SMOKE_LATENT_WIDTH:-16}" \
  --results-dir "${RESULTS_DIR}" \
  --exp-name "${EXP_NAME}" \
  --log-every 1 \
  --ckpt-every 1 \
  2>&1 | tee -a "${RESULTS_DIR}/${EXP_NAME}/launcher.log"
