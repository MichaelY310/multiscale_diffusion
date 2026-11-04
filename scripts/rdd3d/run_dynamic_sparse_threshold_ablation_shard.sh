#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
SHARD="${RDD3D_THRESHOLD_SHARD:?set RDD3D_THRESHOLD_SHARD to 0 or 1}"
CHECKPOINT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/checkpoints/step0015000.pt"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
CONDITIONS="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/conditions_dinov2_vits14"
THRESHOLDS="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/step0015000_thresholds.json"
OUTPUT_ROOT="${ROOT}/results_rdd3d/dynamic_sparse_15k_threshold_ablations_seed20260811"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/15k_threshold_ablations_seed20260811"

mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

run_case() {
    local name="$1"
    shift
    local output="${OUTPUT_ROOT}/${name}"
    echo "===== ${name} ====="
    "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${output}" --steps 8 12 20 --limit 4 \
        --seed 20260811 --weights model "$@"
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${output}" --ground-truth-dir "${LATENTS}" \
        --output "${EVAL_ROOT}/${name}.json" --resolution 32 \
        --stage-resolutions 8 16 32
}

case "${SHARD}" in
    0)
        run_case default --pruning-mode or \
            --mass-thresholds 0.05 0.10 0.25
        run_case lower_fine --pruning-mode or \
            --probability-thresholds 0.1787109375 0.85546875 0.90 \
            --mass-thresholds 0.05 0.10 0.15
        run_case lower_refine --pruning-mode or \
            --probability-thresholds 0.1787109375 0.60 0.90 \
            --mass-thresholds 0.05 0.075 0.15
        run_case halo16 --pruning-mode or \
            --mass-thresholds 0.05 0.10 0.25 --halo-radii 0 1 0
        ;;
    1)
        run_case probability_only --pruning-mode probability \
            --mass-thresholds 0.05 0.10 0.25
        run_case probability_only_relaxed --pruning-mode probability \
            --probability-thresholds 0.15 0.60 0.90 \
            --mass-thresholds 0.05 0.10 0.25
        run_case mass_only --pruning-mode mass \
            --mass-thresholds 0.05 0.10 0.25
        run_case keep16 --pruning-mode or \
            --mass-thresholds 0.05 0.10 0.25 --force-keep-resolutions 16
        ;;
    *)
        echo "Unknown shard: ${SHARD}" >&2
        exit 2
        ;;
esac

echo "RDD3D_THRESHOLD_SHARD_${SHARD}_COMPLETE"
