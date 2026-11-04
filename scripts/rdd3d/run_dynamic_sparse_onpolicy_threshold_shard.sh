#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
SHARD="${RDD3D_ONPOLICY_THRESHOLD_SHARD:?set shard 0 or 1}"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
CONDITIONS="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/conditions_dinov2_vits14"
RUN="${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu"
TRAIN_EVAL="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu"
OUTPUT_ROOT="${ROOT}/results_rdd3d/dynamic_sparse_onpolicy_threshold_seed314159"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/onpolicy_threshold_seed314159"
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

if [[ "${SHARD}" == 0 ]]; then
    STEP=19000
    P8=0.98046875
    P16=0.94921875
elif [[ "${SHARD}" == 1 ]]; then
    STEP=20000
    P8=0.98828125
    P16=0.984375
else
    exit 2
fi
TAG="step$(printf '%07d' "${STEP}")"
CHECKPOINT="${RUN}/checkpoints/${TAG}.pt"
THRESHOLDS="${TRAIN_EVAL}/${TAG}_thresholds.json"

run_case() {
    local variant="$1"
    shift
    local name="${TAG}_${variant}"
    local output="${OUTPUT_ROOT}/${name}"
    "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${output}" --steps 8 12 20 --limit 4 \
        --seed 314159 --weights model "$@"
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${output}" --ground-truth-dir "${LATENTS}" \
        --output "${EVAL_ROOT}/${name}.json" --resolution 32 \
        --stage-resolutions 8 16 32
}

run_case lower_fine --pruning-mode or \
    --probability-thresholds "${P8}" "${P16}" 0.85 \
    --mass-thresholds 0.05 0.10 0.10
run_case lower_refine --pruning-mode or \
    --probability-thresholds "${P8}" 0.75 0.85 \
    --mass-thresholds 0.05 0.075 0.10
run_case halo8 --pruning-mode or --halo-radii 1 0 0 \
    --mass-thresholds 0.05 0.10 0.25
run_case force8_lower_refine --pruning-mode or --force-keep-resolutions 8 \
    --probability-thresholds "${P8}" 0.75 0.85 \
    --mass-thresholds 0.05 0.075 0.10

echo "RDD3D_ONPOLICY_THRESHOLD_SHARD_${SHARD}_COMPLETE"
