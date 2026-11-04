#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
BASE_RUN="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu"
CONDITIONS="${BASE_RUN}/conditions_dinov2_vits14"
RUN_NAME="dynamic_sparse_32asset_replay_ft_w384d12_2gpu"
OUTPUT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"
SEED="${RDD3D_EVAL_SEED:-20260811}"

cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

for STEP in 11000 12000 13000 14000 15000; do
    TAG="step$(printf '%07d' "${STEP}")"
    CHECKPOINT="${OUTPUT}/checkpoints/${TAG}.pt"
    THRESHOLDS="${EVAL_ROOT}/${TAG}_thresholds.json"
    SAMPLES="${OUTPUT}/samples_${TAG}_seed${SEED}"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${SAMPLES}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --limit 4 \
        --seed "${SEED}" --weights model
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${SAMPLES}" --ground-truth-dir "${LATENTS}" \
        --output "${EVAL_ROOT}/${TAG}_seed${SEED}_generated_support.json" \
        --resolution 32 --stage-resolutions 8 16 32
done

echo "RDD3D_NEW_SEED_EVAL_COMPLETE seed=${SEED}"
