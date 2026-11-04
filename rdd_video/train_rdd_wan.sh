#!/usr/bin/env bash
# Fine-tune Wan 2.1-1.3B with the RDD spatiotemporal FM path.
# Every path and expensive setting can be overridden through environment vars.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

PRETRAINED_DIR="${PRETRAINED_DIR:-${REPO_DIR}/pretrained_models/Wan2.1-T2V-1.3B}"
DATA_DIR="${DATA_DIR:-${REPO_DIR}/rdd_video_data/multires_latents_latent}"
RESULTS_DIR="${RESULTS_DIR:-${REPO_DIR}/results_rdd_video}"
EXP_NAME="${EXP_NAME:-rdd_wan_1p3b_joint_projection_5k}"

NUM_GPUS="${NUM_GPUS:-8}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((3 * NUM_GPUS))}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-0}"
MAX_STEPS="${MAX_STEPS:-5000}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
LR="${LR:-1e-5}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"

# Coarsest to finest. Time is paper data-time; Wan receives sigma=1-t.
STAGE_CONFIGS="${STAGE_CONFIGS:-4,4,20;2,2,20;1,1,10}"
STAGE_LENGTHS="${STAGE_LENGTHS:-60,350,590}"
STAGE_SAMPLING="${STAGE_SAMPLING:-proportional}"
DOWNSAMPLE_MODE="${DOWNSAMPLE_MODE:-latent}"
DATA_MODE="${DATA_MODE:-multires}"
DRIFT_SCALE="${DRIFT_SCALE:-1.0}"

if [[ ! -d "${PRETRAINED_DIR}" ]]; then
    echo "ERROR: missing Wan checkpoint directory: ${PRETRAINED_DIR}" >&2
    echo "Run: bash ${SCRIPT_DIR}/setup_wan21.sh" >&2
    exit 2
fi
if [[ "${DATA_MODE}" != "dummy" && ! -d "${DATA_DIR}" ]]; then
    echo "ERROR: missing training data: ${DATA_DIR}" >&2
    echo "Run: bash ${SCRIPT_DIR}/prepare_rdd_wan_data.sh" >&2
    exit 2
fi
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS" >&2
    exit 2
fi

mkdir -p "${RESULTS_DIR}/${EXP_NAME}"
LOG_FILE="${RESULTS_DIR}/${EXP_NAME}/launcher.log"

echo "RDD-Wan training"
echo "  repo=${REPO_DIR}"
echo "  model=${PRETRAINED_DIR}"
echo "  data=${DATA_DIR} (${DATA_MODE}, projection=${DOWNSAMPLE_MODE})"
echo "  stages=${STAGE_CONFIGS}; lengths=${STAGE_LENGTHS}; sampling=${STAGE_SAMPLING}; drift=${DRIFT_SCALE}"
echo "  GPUs=${NUM_GPUS}; global batch=${GLOBAL_BATCH_SIZE}; micro batch/GPU=${MICRO_BATCH_SIZE}; steps=${MAX_STEPS}"
echo "  log=${LOG_FILE}"

cd "${REPO_DIR}"
torchrun --standalone --nproc_per_node="${NUM_GPUS}" \
    rdd_video/train_rdd_wan.py \
    --pretrained-dir "${PRETRAINED_DIR}" \
    --data-dir "${DATA_DIR}" \
    --data-mode "${DATA_MODE}" \
    --downsample-mode "${DOWNSAMPLE_MODE}" \
    --stage-configs "${STAGE_CONFIGS}" \
    --stage-lengths "${STAGE_LENGTHS}" \
    --stage-sampling "${STAGE_SAMPLING}" \
    --drift-scale "${DRIFT_SCALE}" \
    --lr "${LR}" \
    --global-batch-size "${GLOBAL_BATCH_SIZE}" \
    --micro-batch-size "${MICRO_BATCH_SIZE}" \
    --max-steps "${MAX_STEPS}" \
    --warmup-steps "${WARMUP_STEPS}" \
    --exp-name "${EXP_NAME}" \
    --results-dir "${RESULTS_DIR}" \
    --log-every 50 \
    --ckpt-every 1000 \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" 2>&1 | tee -a "${LOG_FILE}"

echo "Checkpoints: ${RESULTS_DIR}/${EXP_NAME}/checkpoints"
