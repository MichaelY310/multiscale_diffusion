#!/usr/bin/env bash
set -euo pipefail

# Sequential policy calibration for hierarchical dynamic pruning.  A threshold
# at stage r changes the candidate distribution at stage 2r, so calibrating all
# stages from one fixed rollout is off-policy.  This script calibrates 8, rolls
# out again, calibrates 16, rolls out again, and finally calibrates 32.

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
CHECKPOINT="${RDD3D_CLOSED_LOOP_CHECKPOINT:-${OUTPUT_ROOT}/onpolicy_replay/checkpoints/step0029000.pt}"
CLEAN_THRESHOLDS="${RDD3D_CLOSED_LOOP_CLEAN_THRESHOLDS:-${EVAL_ROOT}/onpolicy_step0029000_clean_thresholds.json}"
WORK="${OUTPUT_ROOT}/closed_loop_calibration"
WORK="${RDD3D_CLOSED_LOOP_WORK:-${WORK}}"
OUTPUT_STEM="${RDD3D_CLOSED_LOOP_OUTPUT_STEM:-onpolicy_step0029000_closed_loop}"
TAG_PREFIX="${RDD3D_CLOSED_LOOP_TAG_PREFIX:-onpolicy_closed_loop}"
# Use '-' rather than ':-' so a wrapper can intentionally request no suffix.
TAG_SUFFIX="${RDD3D_CLOSED_LOOP_TAG_SUFFIX-_cal_step0029000}"
LABELS="${RDD3D_CLOSED_LOOP_LABELS:-iou f2}"

# Optional resolution-dispatched inference, e.g. a strong 8^3 checkpoint plus
# a refinement checkpoint adapted to its generated parent distribution.  The
# values are whitespace-separated because all repository paths are space-free.
STAGE_ARGS=()
STAGE_CHECKPOINTS=()
if [[ -n "${RDD3D_CLOSED_LOOP_STAGE_CHECKPOINTS:-}" ]]; then
    read -r -a STAGE_CHECKPOINTS <<<"${RDD3D_CLOSED_LOOP_STAGE_CHECKPOINTS}"
    if [[ "${#STAGE_CHECKPOINTS[@]}" -ne 3 ]]; then
        echo "RDD3D_CLOSED_LOOP_STAGE_CHECKPOINTS must contain 3 paths (8/16/32)." >&2
        exit 2
    fi
    if [[ -n "${RDD3D_CLOSED_LOOP_STAGE_WEIGHTS:-}" ]]; then
        read -r -a STAGE_WEIGHTS <<<"${RDD3D_CLOSED_LOOP_STAGE_WEIGHTS}"
    else
        STAGE_WEIGHTS=(model model model)
    fi
    if [[ "${#STAGE_WEIGHTS[@]}" -ne 3 ]]; then
        echo "RDD3D_CLOSED_LOOP_STAGE_WEIGHTS must contain 3 entries." >&2
        exit 2
    fi
    STAGE_ARGS=(--stage-checkpoints "${STAGE_CHECKPOINTS[@]}" --stage-weights "${STAGE_WEIGHTS[@]}")
fi

for REQUIRED in "${PYTHON}" "${TRAIN_LATENTS}" "${HELDOUT_LATENTS}" \
    "${CONDITIONS}" "${CHECKPOINT}" "${CLEAN_THRESHOLDS}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing closed-loop calibration input: ${REQUIRED}" >&2
        exit 2
    fi
done
for REQUIRED in "${STAGE_CHECKPOINTS[@]}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing closed-loop stage checkpoint: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${WORK}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

rollout_train_pair() {
    local label="$1"
    local policy="$2"
    local phase="$3"
    local seed0="$4"
    local seed1="$5"
    local dir0="${WORK}/${label}_${phase}_seed${seed0}"
    local dir1="${WORK}/${label}_${phase}_seed${seed1}"
    mkdir -p "${dir0}" "${dir1}"
    if [[ ! -s "${dir0}/manifest.json" ]]; then
        CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
            --checkpoint "${CHECKPOINT}" "${STAGE_ARGS[@]}" --latent-dir "${TRAIN_LATENTS}" \
            --condition-dir "${CONDITIONS}" --thresholds "${policy}" \
            --output-dir "${dir0}" --steps 8 12 20 \
            --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
            --limit 96 --seed "${seed0}" --weights model \
            >"${ROOT}/logs/${RUN_NAME}_closed_loop_${label}_${phase}_seed${seed0}.log" 2>&1 &
        pid0=$!
    else
        pid0=""
    fi
    if [[ ! -s "${dir1}/manifest.json" ]]; then
        CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
            --checkpoint "${CHECKPOINT}" "${STAGE_ARGS[@]}" --latent-dir "${TRAIN_LATENTS}" \
            --condition-dir "${CONDITIONS}" --thresholds "${policy}" \
            --output-dir "${dir1}" --steps 8 12 20 \
            --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
            --limit 96 --seed "${seed1}" --weights model \
            >"${ROOT}/logs/${RUN_NAME}_closed_loop_${label}_${phase}_seed${seed1}.log" 2>&1 &
        pid1=$!
    else
        pid1=""
    fi
    [[ -z "${pid0}" ]] || wait "${pid0}"
    [[ -z "${pid1}" ]] || wait "${pid1}"
    printf '%s\n%s\n' "${dir0}" "${dir1}"
}

calibrate_report() {
    local label="$1"
    local dir0="$2"
    local dir1="$3"
    local output="$4"
    local objective_args=(--objective iou)
    if [[ "${label}" == "f2" ]]; then
        objective_args=(--objective fbeta --fbeta 2)
    fi
    "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning_replay.py \
        --replay-dirs "${dir0}" "${dir1}" --ground-truth-dir "${TRAIN_LATENTS}" \
        --output "${output}" --fine-resolution 32 --resolutions 8 16 32 \
        "${objective_args[@]}" --limit 96
}

sample_heldout() {
    local tag="$1"
    local policy="$2"
    local seed="$3"
    local gpu="$4"
    local samples="${OUTPUT_ROOT}/samples_${tag}_seed${seed}"
    local report="${EVAL_ROOT}/${tag}_seed${seed}_heldout_support.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" "${STAGE_ARGS[@]}" --latent-dir "${HELDOUT_LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${policy}" \
        --output-dir "${samples}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
        --limit 32 --seed "${seed}" --weights model \
        >"${ROOT}/logs/${RUN_NAME}_${tag}_seed${seed}.log" 2>&1
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${samples}" --ground-truth-dir "${HELDOUT_LATENTS}" \
        --output "${report}" --resolution 32 --stage-resolutions 8 16 32
}

for LABEL in ${LABELS}; do
    if [[ "${LABEL}" == "iou" ]]; then
        INITIAL="${RDD3D_CLOSED_LOOP_INITIAL_IOU:-${EVAL_ROOT}/onpolicy_step0029000_generated_iou_thresholds.json}"
        SEED_INITIAL_A=31301
        SEED_INITIAL_B=31302
        SEED8_A=31501
        SEED8_B=31502
        SEED16_A=31601
        SEED16_B=31602
    else
        INITIAL="${RDD3D_CLOSED_LOOP_INITIAL_F2:-${EVAL_ROOT}/onpolicy_step0029000_generated_f2_thresholds.json}"
        SEED_INITIAL_A=32301
        SEED_INITIAL_B=32302
        SEED8_A=32501
        SEED8_B=32502
        SEED16_A=32601
        SEED16_B=32602
    fi
    if [[ ! -s "${INITIAL}" ]]; then
        mapfile -t INITIAL_DIRS < <(
            rollout_train_pair "${LABEL}" "${CLEAN_THRESHOLDS}" initial "${SEED_INITIAL_A}" "${SEED_INITIAL_B}"
        )
        calibrate_report "${LABEL}" "${INITIAL_DIRS[0]}" "${INITIAL_DIRS[1]}" "${INITIAL}"
    fi

    POLICY8="${WORK}/${LABEL}_policy_after8.json"
    if [[ ! -s "${POLICY8}" ]]; then
        "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
            --base "${CLEAN_THRESHOLDS}" --generated "${INITIAL}" \
            --replace-resolutions 8 --output "${POLICY8}"
    fi
    mapfile -t AFTER8_DIRS < <(
        rollout_train_pair "${LABEL}" "${POLICY8}" after8 "${SEED8_A}" "${SEED8_B}"
    )
    AFTER8_REPORT="${WORK}/${LABEL}_calibration_after8.json"
    if [[ ! -s "${AFTER8_REPORT}" ]]; then
        calibrate_report "${LABEL}" "${AFTER8_DIRS[0]}" "${AFTER8_DIRS[1]}" "${AFTER8_REPORT}"
    fi

    POLICY16="${WORK}/${LABEL}_policy_after16.json"
    if [[ ! -s "${POLICY16}" ]]; then
        "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
            --base "${POLICY8}" --generated "${AFTER8_REPORT}" \
            --replace-resolutions 16 --output "${POLICY16}"
    fi
    mapfile -t AFTER16_DIRS < <(
        rollout_train_pair "${LABEL}" "${POLICY16}" after16 "${SEED16_A}" "${SEED16_B}"
    )
    AFTER16_REPORT="${WORK}/${LABEL}_calibration_after16.json"
    if [[ ! -s "${AFTER16_REPORT}" ]]; then
        calibrate_report "${LABEL}" "${AFTER16_DIRS[0]}" "${AFTER16_DIRS[1]}" "${AFTER16_REPORT}"
    fi

    FINAL_POLICY="${EVAL_ROOT}/${OUTPUT_STEM}_${LABEL}_thresholds.json"
    if [[ ! -s "${FINAL_POLICY}" ]]; then
        "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
            --base "${POLICY16}" --generated "${AFTER16_REPORT}" \
            --replace-resolutions 32 --output "${FINAL_POLICY}"
    fi

    TAG="${TAG_PREFIX}_${LABEL}${TAG_SUFFIX}"
    sample_heldout "${TAG}" "${FINAL_POLICY}" 314159 0 &
    pid0=$!
    sample_heldout "${TAG}" "${FINAL_POLICY}" 271828 1 &
    pid1=$!
    wait "${pid0}"
    wait "${pid1}"
done

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/formal_summary.json" \
    --output-markdown "${EVAL_ROOT}/formal_summary.md"

echo "RDD3D_CLOSED_LOOP_CALIBRATION_COMPLETE"
echo "Summary: ${EVAL_ROOT}/formal_summary.md"
