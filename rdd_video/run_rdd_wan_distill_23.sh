#!/usr/bin/env bash
# Launch RDD-aware 2/2/1 distillation on physical GPUs 2 and 3.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
VIDEO_ENV_DIR="${VIDEO_ENV_DIR:-${REPO_DIR}/.venv-rdd-video}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/rdd_video_data_paper/multires_latents_latent_wan14b_80k}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results_rdd_video_distill}"
EXP_NAME="${EXP_NAME:-rdd_wan_dmd2_221_exact}"
LOG_DIR="${RESULTS_DIR}/${EXP_NAME}"

if [[ ! -x "${VIDEO_ENV_DIR}/bin/python" ]]; then
  echo "ERROR: missing video environment: ${VIDEO_ENV_DIR}" >&2
  echo "Run: bash ${SCRIPT_DIR}/setup_rdd_video_env.sh" >&2
  exit 2
fi
if [[ -z "${TEACHER_CHECKPOINT}" || ! -f "${TEACHER_CHECKPOINT}" ]]; then
  echo "ERROR: set TEACHER_CHECKPOINT to a converged RDD-Wan checkpoint." >&2
  echo "The original Wan checkpoint is not an RDD teacher and is rejected." >&2
  exit 3
fi
if [[ ! -d "${DATA_DIR}" ]]; then
  echo "ERROR: missing cached multi-resolution training data: ${DATA_DIR}" >&2
  exit 4
fi

mkdir -p "${LOG_DIR}"
source "${VIDEO_ENV_DIR}/bin/activate"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PYTHONPATH="${REPO_DIR}/Wan2.1:${REPO_DIR}/rdd_video:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

echo "Physical GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "RDD teacher: ${TEACHER_CHECKPOINT}"
echo "Student schedule: 2/2/1 (five conditional NFE after CFG distillation)"
echo "Log: ${LOG_DIR}/launcher.log"

RESUME_ARGS=()
if [[ -n "${RESUME:-}" ]]; then
  RESUME_ARGS=(--resume "${RESUME}")
fi

torchrun --standalone --nproc_per_node=2 "${SCRIPT_DIR}/distill_rdd_wan.py" \
  --base-model-dir "${BASE_MODEL_DIR}" \
  --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
  --data-dir "${DATA_DIR}" \
  --data-mode multires \
  --stage-configs "${STAGE_CONFIGS:-4,4,2;2,2,2;1,1,1}" \
  --stage-lengths "${STAGE_LENGTHS:-60,350,590}" \
  --student-steps "${STUDENT_STEPS:-2,2,1}" \
  --trajectory-steps "${TRAJECTORY_STEPS:-1000}" \
  --max-steps "${MAX_STEPS:-5000}" \
  --teacher-substeps "${TEACHER_SUBSTEPS:-8}" \
  --teacher-solver "${TEACHER_SOLVER:-heun}" \
  --teacher-cfg-scale "${TEACHER_CFG_SCALE:-5.0}" \
  --path-correction "${PATH_CORRECTION:-exact}" \
  --teacher-loss-weight "${TEACHER_LOSS_WEIGHT:-0.01}" \
  --generator-update-interval "${GENERATOR_UPDATE_INTERVAL:-5}" \
  --lora-rank "${LORA_RANK:-128}" \
  --lora-alpha "${LORA_ALPHA:-128}" \
  --critic-lora-rank "${CRITIC_LORA_RANK:-128}" \
  --critic-lora-alpha "${CRITIC_LORA_ALPHA:-128}" \
  --student-lr "${STUDENT_LR:-1e-5}" \
  --critic-lr "${CRITIC_LR:-1e-5}" \
  --micro-batch-size "${MICRO_BATCH_SIZE:-1}" \
  --grad-accum-steps "${GRAD_ACCUM_STEPS:-6}" \
  --num-workers "${NUM_WORKERS:-2}" \
  --results-dir "${RESULTS_DIR}" \
  --exp-name "${EXP_NAME}" \
  --log-every "${LOG_EVERY:-10}" \
  --ckpt-every "${CKPT_EVERY:-500}" \
  "${RESUME_ARGS[@]}" \
  2>&1 | tee -a "${LOG_DIR}/launcher.log"
