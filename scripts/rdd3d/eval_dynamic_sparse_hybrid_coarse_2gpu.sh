#!/usr/bin/env bash
set -euo pipefail

# Diagnostic for stage interference: preserve the strong zero-shot 19K coarse
# generator at 8^3 and dispatch 16^3/32^3 refinement to the formal 29K model.
# No additional training and no held-out threshold tuning are performed.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SPLIT_ROOT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_96_32_seed98317}"
HELDOUT_LATENTS="${SPLIT_ROOT}/heldout_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"
RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

COARSE_CHECKPOINT="${RDD3D_HYBRID_COARSE_CHECKPOINT:-${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu/checkpoints/step0019000.pt}"
REFINE_CHECKPOINT="${RDD3D_HYBRID_REFINE_CHECKPOINT:-${OUTPUT_ROOT}/onpolicy_replay/checkpoints/step0029000.pt}"
COARSE_THRESHOLDS="${RDD3D_HYBRID_COARSE_THRESHOLDS:-${EVAL_ROOT}/zeroshot_step0019000_thresholds.json}"
REFINE_THRESHOLDS="${RDD3D_HYBRID_REFINE_THRESHOLDS:-${EVAL_ROOT}/onpolicy_step0029000_clean_thresholds.json}"
HYBRID_THRESHOLDS="${EVAL_ROOT}/hybrid19coarse_29refine_clean_thresholds.json"
TAG="hybrid19coarse_29refine_clean_cal"

for REQUIRED in "${PYTHON}" "${HELDOUT_LATENTS}" "${CONDITIONS}" \
    "${COARSE_CHECKPOINT}" "${REFINE_CHECKPOINT}" \
    "${COARSE_THRESHOLDS}" "${REFINE_THRESHOLDS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing hybrid input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

if [[ ! -s "${HYBRID_THRESHOLDS}" ]]; then
    "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
        --base "${REFINE_THRESHOLDS}" --generated "${COARSE_THRESHOLDS}" \
        --replace-resolutions 8 --output "${HYBRID_THRESHOLDS}"
fi

sample_one() {
    local seed="$1"
    local gpu="$2"
    local samples="${OUTPUT_ROOT}/samples_${TAG}_seed${seed}"
    local report="${EVAL_ROOT}/${TAG}_seed${seed}_heldout_support.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${REFINE_CHECKPOINT}" \
        --stage-checkpoints "${COARSE_CHECKPOINT}" "${REFINE_CHECKPOINT}" "${REFINE_CHECKPOINT}" \
        --stage-weights model model model \
        --latent-dir "${HELDOUT_LATENTS}" --condition-dir "${CONDITIONS}" \
        --thresholds "${HYBRID_THRESHOLDS}" --output-dir "${samples}" \
        --steps 8 12 20 --mass-thresholds 0.05 0.10 0.25 \
        --pruning-mode or --limit 32 --seed "${seed}" --weights model \
        >"${ROOT}/logs/${RUN_NAME}_${TAG}_seed${seed}.log" 2>&1
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${samples}" --ground-truth-dir "${HELDOUT_LATENTS}" \
        --output "${report}" --resolution 32 --stage-resolutions 8 16 32
}

sample_one 314159 0 &
pid0=$!
sample_one 271828 1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/formal_summary.json" \
    --output-markdown "${EVAL_ROOT}/formal_summary.md"
echo "RDD3D_HYBRID_COARSE_COMPLETE"
