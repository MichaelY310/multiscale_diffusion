#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
SHARD="${RDD3D_ABLATION_SHARD:?set RDD3D_ABLATION_SHARD to 0 or 1}"
CHECKPOINT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/checkpoints/step0010000.pt"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
CONDITIONS="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/conditions_dinov2_vits14"
THRESHOLDS="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_gate_w384d12_2gpu_thresholds.json"
OUTPUT_ROOT="${ROOT}/results_rdd3d/dynamic_sparse_10k_sampling_ablations"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/10k_sampling_ablations"

mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
exec > >(tee -a "${ROOT}/logs/rdd3d_ablation_${SHARD}.log") 2>&1
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
        --output-dir "${output}" --mass-thresholds 0.05 0.10 0.25 \
        --limit 4 --weights model "$@"
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${output}" --ground-truth-dir "${LATENTS}" \
        --output "${EVAL_ROOT}/${name}.json" --resolution 32 \
        --stage-resolutions 8 16 32
}

case "${SHARD}" in
    0)
        run_case baseline_diag --solver euler --steps 8 12 20
        run_case euler_more_nfe --solver euler --steps 32 24 40
        run_case dense8_more_nfe --solver euler --steps 32 24 40 \
            --force-keep-resolutions 8
        ;;
    1)
        run_case halo8_r1 --solver euler --steps 8 12 20 \
            --halo-radii 1 0 0
        run_case heun_base --solver heun --steps 8 12 20
        run_case dense8_heun --solver heun --steps 16 16 24 \
            --force-keep-resolutions 8
        ;;
    *)
        echo "Unknown shard: ${SHARD}" >&2
        exit 2
        ;;
esac

echo "ABLATION_SHARD_${SHARD}_COMPLETE"
