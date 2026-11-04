#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != rdd && "$1" != flat ) ]]; then
    echo "Usage: $0 {rdd|flat}" >&2
    exit 2
fi
MODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
GPU_LIST="${RDD3D_TRAIN_GPUS:-4,5}"
NUM_GPUS="${RDD3D_NUM_GPUS:-2}"
MAX_STEPS="${RDD3D_MAX_STEPS:-2000}"
BATCH_PER_GPU="${RDD3D_BATCH_PER_GPU:-1}"
DATALOADER_WORKERS="${RDD3D_DATALOADER_WORKERS:-0}"
IMAGE_ENCODER="${RDD3D_IMAGE_ENCODER:-dinov3}"
COMPILE_THREADS="${RDD3D_COMPILE_THREADS:-1}"
TRYRUN="${RDD3D_TRYRUN:-0}"
MODEL_RESOLUTION="${RDD3D_MODEL_RESOLUTION:-512}"
RUN_NAME="${RDD3D_RUN_NAME:-pilot}"
if [[ -n "${RDD3D_MASTER_PORT:-}" ]]; then
    MASTER_PORT="${RDD3D_MASTER_PORT}"
elif [[ "${MODE}" == rdd ]]; then
    MASTER_PORT=12355
else
    MASTER_PORT=12356
fi
SUFFIX="train"
if [[ "${TRYRUN}" == 1 ]]; then
    SUFFIX="tryrun"
fi
LOG="${ROOT}/logs/trellis2_${MODE}_${MODEL_RESOLUTION}_${RUN_NAME}_${SUFFIX}.log"
SESSION="rdd3d_${MODE}_${MODEL_RESOLUTION}_${RUN_NAME}_${SUFFIX}"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

TMUX_ENV=(
    -e "RDD3D_TRAIN_GPUS=${GPU_LIST}"
    -e "RDD3D_NUM_GPUS=${NUM_GPUS}"
    -e "RDD3D_MAX_STEPS=${MAX_STEPS}"
    -e "RDD3D_BATCH_PER_GPU=${BATCH_PER_GPU}"
    -e "RDD3D_DATALOADER_WORKERS=${DATALOADER_WORKERS}"
    -e "RDD3D_IMAGE_ENCODER=${IMAGE_ENCODER}"
    -e "RDD3D_COMPILE_THREADS=${COMPILE_THREADS}"
    -e "RDD3D_TRYRUN=${TRYRUN}"
    -e "RDD3D_MASTER_PORT=${MASTER_PORT}"
    -e "RDD3D_MODEL_RESOLUTION=${MODEL_RESOLUTION}"
    -e "RDD3D_RUN_NAME=${RUN_NAME}"
)
for NAME in RDD3D_DATA_ROOT RDD3D_DATASET_NAME RDD3D_OUTPUT RDD3D_LR RDD3D_EMA_RATE RDD3D_MIN_AESTHETIC_SCORE; do
    if [[ -n "${!NAME:-}" ]]; then
        TMUX_ENV+=(-e "${NAME}=${!NAME}")
    fi
done

tmux new-session -d -s "${SESSION}" "${TMUX_ENV[@]}" \
    "bash -lc 'cd \"${ROOT}\" && bash scripts/rdd3d/train_trellis2_finetune_payload.sh \"${MODE}\" 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION} on CUDA_VISIBLE_DEVICES=${GPU_LIST}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
