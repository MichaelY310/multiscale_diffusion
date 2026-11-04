#!/usr/bin/env bash
set -euo pipefail

# Post-gate diagnostic: fit only the 385-parameter occupancy head on sampler
# endpoint states.  The frozen denoiser/velocity path remains the exact 29K
# formal model.  All fitting and threshold calibration use train assets only;
# held-out assets are touched only by the final two-seed evaluation.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SPLIT_ROOT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_96_32_seed98317}"
TRAIN_LATENTS="${SPLIT_ROOT}/train_latents"
HELDOUT_LATENTS="${SPLIT_ROOT}/heldout_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"

FORMAL_RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
FORMAL_OUTPUT="${ROOT}/results_rdd3d/${FORMAL_RUN_NAME}"
FORMAL_EVAL="${ROOT}/eval_results/rdd3d_dynamic_sparse/${FORMAL_RUN_NAME}"
BASE_CHECKPOINT="${RDD3D_ENDPOINT_BASE_CHECKPOINT:-${FORMAL_OUTPUT}/onpolicy_replay/checkpoints/step0029000.pt}"
BASE_THRESHOLDS="${RDD3D_ENDPOINT_BASE_THRESHOLDS:-${FORMAL_EVAL}/onpolicy_step0029000_clean_thresholds.json}"

RUN_NAME="${FORMAL_RUN_NAME}_endpoint_head"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"
FIT_A="${OUTPUT_ROOT}/fit_replay_seed97531"
FIT_B="${OUTPUT_ROOT}/fit_replay_seed86420"
HEAD_CHECKPOINT="${OUTPUT_ROOT}/endpoint_head/checkpoint.pt"
HEAD_REPORT="${EVAL_ROOT}/endpoint_head_training.json"
HEAD_CLEAN_THRESHOLDS="${EVAL_ROOT}/endpoint_head_clean_thresholds.json"
CALIB_A="${OUTPUT_ROOT}/calibration_seed13579"
CALIB_B="${OUTPUT_ROOT}/calibration_seed24680"
IOU_THRESHOLDS="${EVAL_ROOT}/endpoint_head_generated_iou_thresholds.json"
F2_THRESHOLDS="${EVAL_ROOT}/endpoint_head_generated_f2_thresholds.json"

for REQUIRED in "${PYTHON}" "${TRAIN_LATENTS}" "${HELDOUT_LATENTS}" \
    "${CONDITIONS}" "${BASE_CHECKPOINT}" "${BASE_THRESHOLDS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing endpoint-head input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

sample_train_replay() {
    local output="$1"
    local seed="$2"
    local gpu="$3"
    local log="$4"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${BASE_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${BASE_THRESHOLDS}" \
        --output-dir "${output}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
        --limit 96 --seed "${seed}" --weights model >"${log}" 2>&1
}

echo "===== Endpoint head 1/5: two train-only endpoint replay trajectories ====="
if [[ ! -s "${FIT_A}/manifest.json" || ! -s "${FIT_B}/manifest.json" ]]; then
    sample_train_replay "${FIT_A}" 97531 0 "${ROOT}/logs/${RUN_NAME}_fit_seed97531.log" &
    pid0=$!
    sample_train_replay "${FIT_B}" 86420 1 "${ROOT}/logs/${RUN_NAME}_fit_seed86420.log" &
    pid1=$!
    wait "${pid0}"
    wait "${pid1}"
fi

echo "===== Endpoint head 2/5: frozen-denoiser occupancy-head fitting (GPU 0) ====="
if [[ ! -s "${HEAD_CHECKPOINT}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" \
        RDD_DiT_3D/finetune_dynamic_sparse_occupancy_replay.py \
        --checkpoint "${BASE_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --replay-dirs "${FIT_A}" "${FIT_B}" \
        --output "${HEAD_CHECKPOINT}" --report "${HEAD_REPORT}" \
        --weights model --steps 2000 --lr 5e-4 --positive-weight 3 \
        --stage-probabilities 0.50 0.25 0.25 --seed 20260812 --amp bf16 \
        >"${ROOT}/logs/${RUN_NAME}_head_fit.log" 2>&1
fi

echo "===== Endpoint head 3/5: train-only clean and generated calibration ====="
if [[ ! -s "${HEAD_CLEAN_THRESHOLDS}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${HEAD_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${HEAD_CLEAN_THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit 96 --weights model
fi

sample_head_calibration() {
    local output="$1"
    local seed="$2"
    local gpu="$3"
    local log="$4"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${HEAD_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${HEAD_CLEAN_THRESHOLDS}" \
        --output-dir "${output}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
        --limit 96 --seed "${seed}" --weights model >"${log}" 2>&1
}

if [[ ! -s "${CALIB_A}/manifest.json" || ! -s "${CALIB_B}/manifest.json" ]]; then
    sample_head_calibration "${CALIB_A}" 13579 0 "${ROOT}/logs/${RUN_NAME}_calib_seed13579.log" &
    pid0=$!
    sample_head_calibration "${CALIB_B}" 24680 1 "${ROOT}/logs/${RUN_NAME}_calib_seed24680.log" &
    pid1=$!
    wait "${pid0}"
    wait "${pid1}"
fi

if [[ ! -s "${IOU_THRESHOLDS}" ]]; then
    "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning_replay.py \
        --replay-dirs "${CALIB_A}" "${CALIB_B}" --ground-truth-dir "${TRAIN_LATENTS}" \
        --output "${IOU_THRESHOLDS}" --fine-resolution 32 --resolutions 8 16 32 \
        --objective iou --limit 96
fi
if [[ ! -s "${F2_THRESHOLDS}" ]]; then
    "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning_replay.py \
        --replay-dirs "${CALIB_A}" "${CALIB_B}" --ground-truth-dir "${TRAIN_LATENTS}" \
        --output "${F2_THRESHOLDS}" --fine-resolution 32 --resolutions 8 16 32 \
        --objective fbeta --fbeta 2 --limit 96
fi

sample_heldout() {
    local tag="$1"
    local thresholds="$2"
    local seed="$3"
    local gpu="$4"
    local samples="${OUTPUT_ROOT}/samples_${tag}_seed${seed}"
    local report="${EVAL_ROOT}/${tag}_seed${seed}_heldout_support.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${HEAD_CHECKPOINT}" --latent-dir "${HELDOUT_LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${thresholds}" \
        --output-dir "${samples}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
        --limit 32 --seed "${seed}" --weights model \
        >"${ROOT}/logs/${RUN_NAME}_${tag}_seed${seed}.log" 2>&1
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${samples}" --ground-truth-dir "${HELDOUT_LATENTS}" \
        --output "${report}" --resolution 32 --stage-resolutions 8 16 32
}

evaluate_two_seeds() {
    local tag="$1"
    local thresholds="$2"
    sample_heldout "${tag}" "${thresholds}" 314159 0 &
    pid0=$!
    sample_heldout "${tag}" "${thresholds}" 271828 1 &
    pid1=$!
    wait "${pid0}"
    wait "${pid1}"
}

echo "===== Endpoint head 4/5: two-seed held-out support ====="
evaluate_two_seeds endpoint_head_generated_iou "${IOU_THRESHOLDS}"
evaluate_two_seeds endpoint_head_generated_f2 "${F2_THRESHOLDS}"

echo "===== Endpoint head 5/5: complete ====="
"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/endpoint_head_summary.json" \
    --output-markdown "${EVAL_ROOT}/endpoint_head_summary.md"
echo "RDD3D_ENDPOINT_HEAD_COMPLETE"
echo "Head report: ${HEAD_REPORT}"
echo "IoU held-out: ${EVAL_ROOT}/endpoint_head_generated_iou_seed314159_heldout_support.json"
echo "F2 held-out:  ${EVAL_ROOT}/endpoint_head_generated_f2_seed314159_heldout_support.json"
