#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != rdd && "$1" != flat ) ]]; then
    echo "Usage: $0 {rdd|flat}" >&2
    exit 2
fi
MODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRELLIS="${ROOT}/third_party/TRELLIS.2"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
GPU_LIST="${RDD3D_TRAIN_GPUS:-4,5}"
NUM_GPUS="${RDD3D_NUM_GPUS:-2}"
MASTER_PORT="${RDD3D_MASTER_PORT:-12345}"
DATALOADER_WORKERS="${RDD3D_DATALOADER_WORKERS:-0}"
IMAGE_ENCODER="${RDD3D_IMAGE_ENCODER:-dinov3}"
COMPILE_THREADS="${RDD3D_COMPILE_THREADS:-1}"
MAX_STEPS="${RDD3D_MAX_STEPS:-2000}"
BATCH_PER_GPU="${RDD3D_BATCH_PER_GPU:-1}"
MODEL_RESOLUTION="${RDD3D_MODEL_RESOLUTION:-512}"
RUN_NAME="${RDD3D_RUN_NAME:-pilot}"
MIN_AESTHETIC_SCORE="${RDD3D_MIN_AESTHETIC_SCORE:--1.0}"
EMA_RATE="${RDD3D_EMA_RATE:-0.9999}"
DATASET_NAME="${RDD3D_DATASET_NAME:-ObjaverseXL}"
CHECKPOINT_DIR="${ROOT}/checkpoints/pretrained/trellis2"
CHECKPOINT="${CHECKPOINT_DIR}/slat_flow_img2shape_dit_1_3B_${MODEL_RESOLUTION}_bf16.pt"
CONFIG="${ROOT}/RDD_DiT_3D/configs/trellis2_${MODE}_${MODEL_RESOLUTION}_${RUN_NAME}.json"
OUTPUT="${RDD3D_OUTPUT:-${ROOT}/results_rdd3d/trellis2_${MODE}_${MODEL_RESOLUTION}_${RUN_NAME}}"

case "${MODEL_RESOLUTION}" in
    512)
        UPSTREAM_CONFIG="${TRELLIS}/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16.json"
        DEFAULT_LR=1e-5
        ;;
    1024)
        UPSTREAM_CONFIG="${TRELLIS}/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json"
        DEFAULT_LR=2e-5
        ;;
    *)
        echo "RDD3D_MODEL_RESOLUTION must be 512 or 1024, got ${MODEL_RESOLUTION}" >&2
        exit 2
        ;;
esac
LR="${RDD3D_LR:-${DEFAULT_LR}}"

mkdir -p "${CHECKPOINT_DIR}" "$(dirname "${CONFIG}")" "${ROOT}/logs" "${OUTPUT}"
if [[ ! -f "${CHECKPOINT}" ]]; then
    "${PYTHON}" scripts/rdd3d/download_trellis2_rdd_checkpoints.py \
        --output-dir "${CHECKPOINT_DIR}" --resolution "${MODEL_RESOLUTION}"
fi
"${PYTHON}" scripts/rdd3d/make_trellis2_finetune_config.py \
    --upstream-config "${UPSTREAM_CONFIG}" \
    --output "${CONFIG}" --checkpoint "${CHECKPOINT}" --mode "${MODE}" \
    --max-steps "${MAX_STEPS}" --batch-size-per-gpu "${BATCH_PER_GPU}" \
    --lr "${LR}" --ema-rate "${EMA_RATE}" \
    --min-aesthetic-score "${MIN_AESTHETIC_SCORE}" \
    --image-encoder "${IMAGE_ENCODER}"

DATA_JSON="{\"${DATASET_NAME}\":{\"base\":\"${DATA_ROOT}\",\"shape_latent\":\"${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${MODEL_RESOLUTION}\",\"render_cond\":\"${DATA_ROOT}/renders_cond\"}}"
if [[ ! -d "${DATA_ROOT}/renders_cond" ]]; then
    echo "Missing ${DATA_ROOT}/renders_cond; prepare image conditions before training." >&2
    exit 2
fi

ENTRY="${ROOT}/RDD_DiT_3D/train_trellis2_runtime.py"

EXTRA_ARGS=()
if [[ "${RDD3D_TRYRUN:-0}" == 1 ]]; then
    EXTRA_ARGS+=(--tryrun)
fi

cd "${ROOT}"
export CUDA_VISIBLE_DEVICES="${GPU_LIST}"
export SPARSE_CONV_BACKEND=flex_gemm
export ATTN_BACKEND=xformers
export TORCHINDUCTOR_COMPILE_THREADS="${COMPILE_THREADS}"
export PYTHONPATH="${TRELLIS}:${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"
exec "${PYTHON}" "${ENTRY}" \
    --trainer-mode "${MODE}" \
    --config "${CONFIG}" --output_dir "${OUTPUT}" \
    --data_dir "${DATA_JSON}" --num_gpus "${NUM_GPUS}" --auto_retry 0 \
    --master_port "${MASTER_PORT}" \
    "${EXTRA_ARGS[@]}"
