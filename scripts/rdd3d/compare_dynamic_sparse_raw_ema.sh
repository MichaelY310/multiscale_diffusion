#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
RUN_NAME="${RDD3D_DYNAMIC_RUN_NAME:-dynamic_sparse_32asset_2k}"
OUTPUT="${RDD3D_DYNAMIC_OUTPUT:-${ROOT}/results_rdd3d/${RUN_NAME}}"
DATA_ROOT="${RDD3D_DYNAMIC_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
LATENTS="${RDD3D_DYNAMIC_LATENTS:-${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512}"
CONDITIONS="${RDD3D_DYNAMIC_CONDITIONS:-${OUTPUT}/conditions_dinov2_vits14}"
CHECKPOINT="${RDD3D_DYNAMIC_CHECKPOINT:-${OUTPUT}/checkpoints/step0002000.pt}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse"
LIMIT="${RDD3D_DYNAMIC_EVAL_LIMIT:-4}"
CALIBRATION_LIMIT="${RDD3D_DYNAMIC_CALIBRATION_LIMIT:-32}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 2
fi
if [[ ! -f "${CHECKPOINT}" ]]; then
    echo "Missing checkpoint: ${CHECKPOINT}" >&2
    exit 2
fi
mkdir -p "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

for WEIGHTS in ema model; do
    if [[ "${WEIGHTS}" == "ema" ]]; then
        SUFFIX="ema"
    else
        SUFFIX="raw"
    fi
    THRESHOLDS="${EVAL_ROOT}/${RUN_NAME}_thresholds_${SUFFIX}.json"
    SAMPLES="${OUTPUT}/samples_dynamic_${SUFFIX}"
    METRICS="${EVAL_ROOT}/${RUN_NAME}_generated_support_${SUFFIX}.json"

    "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit "${CALIBRATION_LIMIT}" \
        --weights "${WEIGHTS}"

    "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${SAMPLES}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --limit "${LIMIT}" \
        --weights "${WEIGHTS}"

    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${SAMPLES}" --ground-truth-dir "${LATENTS}" \
        --output "${METRICS}" --resolution 32
done

echo "raw/EMA diagnostic complete"
echo "EMA metrics: ${EVAL_ROOT}/${RUN_NAME}_generated_support_ema.json"
echo "raw metrics: ${EVAL_ROOT}/${RUN_NAME}_generated_support_raw.json"
