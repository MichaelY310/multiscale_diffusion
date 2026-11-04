#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
CONDITIONS="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/conditions_dinov2_vits14"
CHECKPOINT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/checkpoints/step0015000.pt"
THRESHOLDS="${1:-${RDD3D_GENERATED_THRESHOLDS:-/tmp/rdd3d_replay_calibration_iou.json}}"
TAG="${2:-${RDD3D_GENERATED_CALIBRATION_TAG:-generated_iou_probability_seed314159}}"
PRUNING_MODE="${3:-${RDD3D_GENERATED_PRUNING_MODE:-probability}}"
OUTPUT="${ROOT}/results_rdd3d/dynamic_sparse_15k_${TAG}"
REPORT="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_15k_${TAG}.json"

for REQUIRED in "${PYTHON}" "${LATENTS}" "${CONDITIONS}" "${CHECKPOINT}" "${THRESHOLDS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing required input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT}" "$(dirname "${REPORT}")"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
    --output-dir "${OUTPUT}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit 4 \
    --pruning-mode "${PRUNING_MODE}" --seed 314159 --weights model

"${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
    --prediction-dir "${OUTPUT}" --ground-truth-dir "${LATENTS}" \
    --output "${REPORT}" --resolution 32 --stage-resolutions 8 16 32

echo "RDD3D_GENERATED_CALIBRATION_PILOT_COMPLETE"
echo "${REPORT}"
