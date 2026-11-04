#!/usr/bin/env bash
set -euo pipefail

# Adapt only the refinement model to generated parents from the strong 19K
# coarse model.  Inference remains stage-dispatched: 19K at 8^3 and the adapted
# checkpoint at 16^3/32^3.  The coarse model is never updated.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SPLIT_ROOT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_96_32_seed98317}"
TRAIN_LATENTS="${SPLIT_ROOT}/train_latents"
HELDOUT_LATENTS="${SPLIT_ROOT}/heldout_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"
RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

COARSE_CHECKPOINT="${RDD3D_INTERFACE_COARSE_CHECKPOINT:-${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu/checkpoints/step0019000.pt}"
INIT_REFINE_CHECKPOINT="${RDD3D_INTERFACE_REFINE_CHECKPOINT:-${OUTPUT_ROOT}/onpolicy_replay/checkpoints/step0029000.pt}"
COARSE_THRESHOLDS="${RDD3D_INTERFACE_COARSE_THRESHOLDS:-${EVAL_ROOT}/zeroshot_step0019000_thresholds.json}"
INIT_REFINE_THRESHOLDS="${RDD3D_INTERFACE_REFINE_THRESHOLDS:-${EVAL_ROOT}/onpolicy_step0029000_clean_thresholds.json}"
INIT_HYBRID_THRESHOLDS="${EVAL_ROOT}/hybrid19coarse_29refine_clean_thresholds.json"

ADAPT_ROOT="${OUTPUT_ROOT}/stage_interface_19coarse_refine"
REPLAY_A="${ADAPT_ROOT}/replay_normal_seed551001"
REPLAY_B="${ADAPT_ROOT}/replay_force8_seed551002"
TRAIN_RUN="${ADAPT_ROOT}/refine_adapt"
FINAL_CHECKPOINT="${TRAIN_RUN}/checkpoints/step0034000.pt"
REFINE_CLEAN_THRESHOLDS="${EVAL_ROOT}/interface19coarse_refine34k_clean_raw.json"
FINAL_THRESHOLDS="${EVAL_ROOT}/interface19coarse_refine34k_clean_thresholds.json"
TAG="interface19coarse_refine34k_clean_cal"

for REQUIRED in "${PYTHON}" "${TRAIN_LATENTS}" "${HELDOUT_LATENTS}" "${CONDITIONS}" \
    "${COARSE_CHECKPOINT}" "${INIT_REFINE_CHECKPOINT}" \
    "${COARSE_THRESHOLDS}" "${INIT_REFINE_THRESHOLDS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing stage-interface input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${ADAPT_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

if [[ ! -s "${INIT_HYBRID_THRESHOLDS}" ]]; then
    "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
        --base "${INIT_REFINE_THRESHOLDS}" --generated "${COARSE_THRESHOLDS}" \
        --replace-resolutions 8 --output "${INIT_HYBRID_THRESHOLDS}"
fi

sample_replay() {
    local output="$1"
    local seed="$2"
    local gpu="$3"
    local force8="$4"
    local log="$5"
    local extra=()
    if [[ "${force8}" == "yes" ]]; then
        extra=(--force-keep-resolutions 8 --pruning-mode probability)
    else
        extra=(--pruning-mode or)
    fi
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${INIT_REFINE_CHECKPOINT}" \
        --stage-checkpoints "${COARSE_CHECKPOINT}" "${INIT_REFINE_CHECKPOINT}" "${INIT_REFINE_CHECKPOINT}" \
        --stage-weights model model model \
        --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
        --thresholds "${INIT_HYBRID_THRESHOLDS}" --output-dir "${output}" \
        --steps 8 12 20 --mass-thresholds 0.05 0.10 0.25 \
        --limit 96 --seed "${seed}" --weights model "${extra[@]}" >"${log}" 2>&1
}

echo "===== Interface 1/4: hybrid generated-parent replay (GPUs 0,1) ====="
if [[ ! -s "${REPLAY_A}/manifest.json" || ! -s "${REPLAY_B}/manifest.json" ]]; then
    sample_replay "${REPLAY_A}" 551001 0 no "${ROOT}/logs/${RUN_NAME}_interface_replay_normal.log" &
    pid0=$!
    sample_replay "${REPLAY_B}" 551002 1 yes "${ROOT}/logs/${RUN_NAME}_interface_replay_force8.log" &
    pid1=$!
    wait "${pid0}"
    wait "${pid1}"
fi

echo "===== Interface 2/4: refine-only generated-parent adaptation (GPUs 0,1) ====="
if [[ ! -s "${FINAL_CHECKPOINT}" ]]; then
    CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node 2 RDD_DiT_3D/train_dynamic_sparse_rdd.py \
        --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
        --output "${TRAIN_RUN}" --resume "${INIT_REFINE_CHECKPOINT}" \
        --resume-weights model --reset-optimizer \
        --fine-resolution 32 --resolutions 8 16 32 \
        --time-fractions 0.5 0.3 0.2 --max-assets 96 \
        --steps 5000 --width 384 --depth 12 --lr 1e-5 \
        --stage-probabilities 0 0.40 0.60 \
        --generated-parent-dirs "${REPLAY_A}" "${REPLAY_B}" \
        --generated-parent-probability 1.0 \
        --generated-endpoint-time-probability 0.50 \
        --generated-parent-velocity-weight 0.75 \
        --ema-rate 0.999 --occupancy-loss-weight 0.75 --positive-weight 3.0 \
        --negative-parent-ratio 0 --endpoint-time-probability 0.25 \
        --mass-velocity-weight 4 --save-every 1000 --log-every 25 --amp bf16
fi

echo "===== Interface 3/4: train-only clean threshold composition ====="
if [[ ! -s "${REFINE_CLEAN_THRESHOLDS}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${FINAL_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${REFINE_CLEAN_THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit 96 --weights model
fi
if [[ ! -s "${FINAL_THRESHOLDS}" ]]; then
    "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
        --base "${REFINE_CLEAN_THRESHOLDS}" --generated "${COARSE_THRESHOLDS}" \
        --replace-resolutions 8 --output "${FINAL_THRESHOLDS}"
fi

sample_heldout() {
    local seed="$1"
    local gpu="$2"
    local samples="${OUTPUT_ROOT}/samples_${TAG}_seed${seed}"
    local report="${EVAL_ROOT}/${TAG}_seed${seed}_heldout_support.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${FINAL_CHECKPOINT}" \
        --stage-checkpoints "${COARSE_CHECKPOINT}" "${FINAL_CHECKPOINT}" "${FINAL_CHECKPOINT}" \
        --stage-weights model model model \
        --latent-dir "${HELDOUT_LATENTS}" --condition-dir "${CONDITIONS}" \
        --thresholds "${FINAL_THRESHOLDS}" --output-dir "${samples}" \
        --steps 8 12 20 --mass-thresholds 0.05 0.10 0.25 \
        --pruning-mode or --limit 32 --seed "${seed}" --weights model \
        >"${ROOT}/logs/${RUN_NAME}_${TAG}_seed${seed}.log" 2>&1
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${samples}" --ground-truth-dir "${HELDOUT_LATENTS}" \
        --output "${report}" --resolution 32 --stage-resolutions 8 16 32
}

echo "===== Interface 4/4: two-seed held-out clean/OR evaluation ====="
sample_heldout 314159 0 &
pid0=$!
sample_heldout 271828 1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/formal_summary.json" \
    --output-markdown "${EVAL_ROOT}/formal_summary.md"
echo "RDD3D_STAGE_INTERFACE_ADAPT_COMPLETE"
