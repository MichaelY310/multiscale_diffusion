#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_DYNAMIC_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
LATENTS="${RDD3D_DYNAMIC_LATENTS:-${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512}"
RENDERS="${RDD3D_DYNAMIC_RENDERS:-${DATA_ROOT}/renders_cond}"
RUN_NAME="${RDD3D_DYNAMIC_RUN_NAME:-dynamic_sparse_32asset_gate_v2}"
OUTPUT="${RDD3D_DYNAMIC_OUTPUT:-${ROOT}/results_rdd3d/${RUN_NAME}}"
CONDITIONS="${OUTPUT}/conditions_dinov2_vits14"
MAX_ASSETS="${RDD3D_DYNAMIC_MAX_ASSETS:-32}"
STEPS="${RDD3D_DYNAMIC_TRAIN_STEPS:-10000}"
WIDTH="${RDD3D_DYNAMIC_WIDTH:-192}"
DEPTH="${RDD3D_DYNAMIC_DEPTH:-6}"
WEIGHTS="${RDD3D_DYNAMIC_WEIGHTS:-model}"
NUM_GPUS="${RDD3D_DYNAMIC_NUM_GPUS:-1}"
SAVE_EVERY="${RDD3D_DYNAMIC_SAVE_EVERY:-1000}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 2
fi
if [[ ! -d "${LATENTS}" ]]; then
    echo "Missing 512/32^3 SLat directory: ${LATENTS}" >&2
    exit 2
fi
if [[ ! -d "${RENDERS}" ]]; then
    echo "Missing conditioning renders: ${RENDERS}" >&2
    exit 2
fi
mkdir -p "${OUTPUT}" "${ROOT}/eval_results/rdd3d_dynamic_sparse"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

"${PYTHON}" RDD_DiT_3D/analyze_dynamic_sparse_rdd.py \
    --latent-dir "${LATENTS}" \
    --output "${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_oracle.json" \
    --fine-resolution 32 --resolutions 8 16 32 --steps 8 12 20 \
    --limit "${MAX_ASSETS}"

"${PYTHON}" RDD_DiT_3D/precompute_dynamic_sparse_conditions.py \
    --render-dir "${RENDERS}" --latent-dir "${LATENTS}" \
    --output "${CONDITIONS}" --model dinov2_vits14 \
    --limit "${MAX_ASSETS}"

TRAIN_LAUNCHER=("${PYTHON}")
if (( NUM_GPUS > 1 )); then
    TRAIN_LAUNCHER+=( -m torch.distributed.run --standalone --nproc_per_node "${NUM_GPUS}" )
fi

"${TRAIN_LAUNCHER[@]}" RDD_DiT_3D/train_dynamic_sparse_rdd.py \
    --latent-dir "${LATENTS}" --condition-dir "${CONDITIONS}" \
    --output "${OUTPUT}" --fine-resolution 32 --resolutions 8 16 32 \
    --time-fractions 0.5 0.3 0.2 --max-assets "${MAX_ASSETS}" \
    --steps "${STEPS}" --width "${WIDTH}" --depth "${DEPTH}" \
    --lr 2e-4 --ema-rate 0.999 --occupancy-loss-weight 0.25 \
    --positive-weight 8 --negative-parent-ratio 1.0 \
    --endpoint-time-probability 0.2 --mass-velocity-weight 4 \
    --save-every "${SAVE_EVERY}" --log-every 25 --amp bf16

CHECKPOINT="${OUTPUT}/checkpoints/step$(printf '%07d' "${STEPS}").pt"
"${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
    --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" \
    --output "${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_thresholds.json" \
    --target-recall 0.999 0.995 0.99 --limit "${MAX_ASSETS}" \
    --weights "${WEIGHTS}"

SAMPLES="${OUTPUT}/samples_dynamic"
"${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" \
    --thresholds "${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_thresholds.json" \
    --output-dir "${SAMPLES}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit 4 \
    --weights "${WEIGHTS}"

"${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
    --prediction-dir "${SAMPLES}" --ground-truth-dir "${LATENTS}" \
    --output "${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_generated_support.json" \
    --resolution 32

echo "Pilot complete."
echo "Checkpoint: ${CHECKPOINT}"
echo "Oracle stats: ${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_oracle.json"
echo "Thresholds: ${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_thresholds.json"
echo "Generated support: ${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}_generated_support.json"
