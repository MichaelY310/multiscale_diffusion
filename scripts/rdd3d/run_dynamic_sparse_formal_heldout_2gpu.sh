#!/usr/bin/env bash
set -euo pipefail

# Formal, leakage-resistant RDD-3D experiment:
#   1) old 32-asset checkpoint zero-shot on new held-out assets;
#   2) teacher-parent adaptation on new training assets;
#   3) on-policy generated-parent replay fine-tuning;
#   4) two held-out sampling seeds and paired TRELLIS mesh decoding.
# This script never exposes more than GPUs 0 and 1.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SOURCE_LATENTS="${RDD3D_FORMAL_LATENTS:-${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_512}"
RENDERS="${RDD3D_FORMAL_RENDERS:-${DATA_ROOT}/renders_cond}"
PILOT_LATENTS="${RDD3D_PILOT_LATENTS:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512}"
TRAIN_COUNT="${RDD3D_FORMAL_TRAIN_COUNT:-96}"
HELDOUT_COUNT="${RDD3D_FORMAL_HELDOUT_COUNT:-32}"
SPLIT_SEED="${RDD3D_FORMAL_SPLIT_SEED:-98317}"
SPLIT_ROOT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_${TRAIN_COUNT}_${HELDOUT_COUNT}_seed${SPLIT_SEED}}"
TRAIN_LATENTS="${SPLIT_ROOT}/train_latents"
HELDOUT_LATENTS="${SPLIT_ROOT}/heldout_latents"
ALL_LATENTS="${SPLIT_ROOT}/all_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"

INIT_CHECKPOINT="${RDD3D_FORMAL_INIT_CHECKPOINT:-${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu/checkpoints/step0019000.pt}"
INIT_STEP="${RDD3D_FORMAL_INIT_STEP:-19000}"
TEACHER_STEPS="${RDD3D_FORMAL_TEACHER_STEPS:-5000}"
REPLAY_STEPS="${RDD3D_FORMAL_REPLAY_STEPS:-5000}"
TEACHER_FINAL_STEP=$((INIT_STEP + TEACHER_STEPS))
REPLAY_FINAL_STEP=$((TEACHER_FINAL_STEP + REPLAY_STEPS))
RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_${TRAIN_COUNT}train_${HELDOUT_COUNT}heldout_w384d12}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
TEACHER_RUN="${OUTPUT_ROOT}/teacher_adapt"
REPLAY_RUN="${OUTPUT_ROOT}/onpolicy_replay"
REPLAY_A="${OUTPUT_ROOT}/replay_seed424242"
REPLAY_B="${OUTPUT_ROOT}/replay_seed777777"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"
MESH_LIMIT="${RDD3D_FORMAL_MESH_LIMIT:-4}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "Missing Python environment: ${PYTHON}" >&2
    exit 2
fi
for REQUIRED in "${SOURCE_LATENTS}" "${RENDERS}" "${INIT_CHECKPOINT}"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing formal experiment input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${OUTPUT_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"

echo "===== Phase 0/7: deterministic train/held-out split and DINO conditions ====="
SPLIT_ARGS=(
    --latent-dir "${SOURCE_LATENTS}" --render-dir "${RENDERS}"
    --output-root "${SPLIT_ROOT}" --train-count "${TRAIN_COUNT}"
    --heldout-count "${HELDOUT_COUNT}" --seed "${SPLIT_SEED}"
)
if [[ -d "${PILOT_LATENTS}" ]]; then
    SPLIT_ARGS+=(--exclude-latent-dir "${PILOT_LATENTS}")
fi
"${PYTHON}" scripts/rdd3d/build_dynamic_sparse_formal_split.py "${SPLIT_ARGS[@]}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/precompute_dynamic_sparse_conditions.py \
    --render-dir "${RENDERS}" --latent-dir "${ALL_LATENTS}" \
    --output "${CONDITIONS}" --model dinov2_vits14

"${PYTHON}" RDD_DiT_3D/analyze_dynamic_sparse_rdd.py \
    --latent-dir "${TRAIN_LATENTS}" --limit "${TRAIN_COUNT}" \
    --fine-resolution 32 --resolutions 8 16 32 --steps 8 12 20 \
    --output "${EVAL_ROOT}/train_oracle.json"
"${PYTHON}" RDD_DiT_3D/analyze_dynamic_sparse_rdd.py \
    --latent-dir "${HELDOUT_LATENTS}" --limit "${HELDOUT_COUNT}" \
    --fine-resolution 32 --resolutions 8 16 32 --steps 8 12 20 \
    --output "${EVAL_ROOT}/heldout_oracle.json"

calibrate() {
    local checkpoint="$1"
    local output="$2"
    if [[ ! -s "${output}" ]]; then
        CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
            --checkpoint "${checkpoint}" --latent-dir "${TRAIN_LATENTS}" \
            --condition-dir "${CONDITIONS}" --output "${output}" \
            --target-recall 0.999 0.995 0.99 --limit "${TRAIN_COUNT}" --weights model
    fi
}

sample_one_seed() {
    local tag="$1"
    local checkpoint="$2"
    local thresholds="$3"
    local seed="$4"
    local gpu="$5"
    local pruning_mode="${6:-or}"
    local samples="${OUTPUT_ROOT}/samples_${tag}_seed${seed}"
    local report="${EVAL_ROOT}/${tag}_seed${seed}_heldout_support.json"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${checkpoint}" --latent-dir "${HELDOUT_LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${thresholds}" \
        --output-dir "${samples}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --limit "${HELDOUT_COUNT}" \
        --pruning-mode "${pruning_mode}" --seed "${seed}" --weights model \
        >"${ROOT}/logs/${RUN_NAME}_${tag}_seed${seed}_sample.log" 2>&1
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${samples}" --ground-truth-dir "${HELDOUT_LATENTS}" \
        --output "${report}" --resolution 32 --stage-resolutions 8 16 32
}

evaluate_two_seeds() {
    local tag="$1"
    local checkpoint="$2"
    local thresholds="$3"
    local pruning_mode="${4:-or}"
    sample_one_seed "${tag}" "${checkpoint}" "${thresholds}" 314159 0 "${pruning_mode}" &
    local pid0=$!
    sample_one_seed "${tag}" "${checkpoint}" "${thresholds}" 271828 1 "${pruning_mode}" &
    local pid1=$!
    wait "${pid0}"
    wait "${pid1}"
}

echo "===== Phase 1/7: zero-shot held-out baseline ====="
ZERO_THRESHOLDS="${EVAL_ROOT}/zeroshot_step$(printf '%07d' "${INIT_STEP}")_thresholds.json"
calibrate "${INIT_CHECKPOINT}" "${ZERO_THRESHOLDS}"
evaluate_two_seeds "zeroshot_step$(printf '%07d' "${INIT_STEP}")" \
    "${INIT_CHECKPOINT}" "${ZERO_THRESHOLDS}"

echo "===== Phase 2/7: teacher-parent adaptation on training split (GPUs 0,1) ====="
TEACHER_CHECKPOINT="${TEACHER_RUN}/checkpoints/step$(printf '%07d' "${TEACHER_FINAL_STEP}").pt"
if [[ ! -s "${TEACHER_CHECKPOINT}" ]]; then
    CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node 2 RDD_DiT_3D/train_dynamic_sparse_rdd.py \
        --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
        --output "${TEACHER_RUN}" --resume "${INIT_CHECKPOINT}" \
        --resume-weights model --reset-optimizer \
        --fine-resolution 32 --resolutions 8 16 32 \
        --time-fractions 0.5 0.3 0.2 --max-assets "${TRAIN_COUNT}" \
        --steps "${TEACHER_STEPS}" --width 384 --depth 12 --lr 5e-5 \
        --stage-probabilities 0.25 0.30 0.45 \
        --ema-rate 0.999 --occupancy-loss-weight 0.75 --positive-weight 3.0 \
        --negative-parent-ratio 1.0 --endpoint-time-probability 0.25 \
        --mass-velocity-weight 4 --save-every 1000 --log-every 25 --amp bf16
fi

echo "===== Phase 3/7: teacher-adapted held-out evaluation ====="
TEACHER_THRESHOLDS="${EVAL_ROOT}/teacher_step$(printf '%07d' "${TEACHER_FINAL_STEP}")_thresholds.json"
calibrate "${TEACHER_CHECKPOINT}" "${TEACHER_THRESHOLDS}"
evaluate_two_seeds "teacher_step$(printf '%07d' "${TEACHER_FINAL_STEP}")" \
    "${TEACHER_CHECKPOINT}" "${TEACHER_THRESHOLDS}"

echo "===== Phase 4/7: two train-only generated-parent trajectories (GPUs 0,1) ====="
mkdir -p "${REPLAY_A}" "${REPLAY_B}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${TEACHER_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${TEACHER_THRESHOLDS}" \
    --output-dir "${REPLAY_A}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit "${TRAIN_COUNT}" \
    --seed 424242 --weights model \
    >"${ROOT}/logs/${RUN_NAME}_replay_seed424242.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${TEACHER_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${TEACHER_THRESHOLDS}" \
    --output-dir "${REPLAY_B}" --steps 8 12 20 \
    --probability-thresholds 0.15 0.60 0.90 --pruning-mode probability \
    --mass-thresholds 0.05 0.10 0.25 --limit "${TRAIN_COUNT}" \
    --seed 777777 --weights model \
    >"${ROOT}/logs/${RUN_NAME}_replay_seed777777.log" 2>&1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"

echo "===== Phase 5/7: on-policy replay fine-tuning (GPUs 0,1) ====="
REPLAY_CHECKPOINT="${REPLAY_RUN}/checkpoints/step$(printf '%07d' "${REPLAY_FINAL_STEP}").pt"
if [[ ! -s "${REPLAY_CHECKPOINT}" ]]; then
    CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -m torch.distributed.run \
        --standalone --nproc_per_node 2 RDD_DiT_3D/train_dynamic_sparse_rdd.py \
        --latent-dir "${TRAIN_LATENTS}" --condition-dir "${CONDITIONS}" \
        --output "${REPLAY_RUN}" --resume "${TEACHER_CHECKPOINT}" \
        --resume-weights model --reset-optimizer \
        --fine-resolution 32 --resolutions 8 16 32 \
        --time-fractions 0.5 0.3 0.2 --max-assets "${TRAIN_COUNT}" \
        --steps "${REPLAY_STEPS}" --width 384 --depth 12 --lr 2e-5 \
        --stage-probabilities 0.25 0.30 0.45 \
        --generated-parent-dirs "${REPLAY_A}" "${REPLAY_B}" \
        --generated-parent-probability 0.75 \
        --generated-endpoint-time-probability 0.50 \
        --generated-parent-velocity-weight 0.75 \
        --ema-rate 0.999 --occupancy-loss-weight 0.75 --positive-weight 3.0 \
        --negative-parent-ratio 1.0 --endpoint-time-probability 0.25 \
        --mass-velocity-weight 4 --save-every 1000 --log-every 25 --amp bf16
fi

echo "===== Phase 6/7: on-policy calibration and final held-out evaluation ====="
REPLAY_CLEAN_THRESHOLDS="${EVAL_ROOT}/onpolicy_step$(printf '%07d' "${REPLAY_FINAL_STEP}")_clean_thresholds.json"
calibrate "${REPLAY_CHECKPOINT}" "${REPLAY_CLEAN_THRESHOLDS}"
CALIB_A="${OUTPUT_ROOT}/final_calibration_seed13579"
CALIB_B="${OUTPUT_ROOT}/final_calibration_seed24680"
mkdir -p "${CALIB_A}" "${CALIB_B}"
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${REPLAY_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${REPLAY_CLEAN_THRESHOLDS}" \
    --output-dir "${CALIB_A}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit "${TRAIN_COUNT}" \
    --pruning-mode probability --seed 13579 --weights model \
    >"${ROOT}/logs/${RUN_NAME}_final_calibration_seed13579.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${REPLAY_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${REPLAY_CLEAN_THRESHOLDS}" \
    --output-dir "${CALIB_B}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit "${TRAIN_COUNT}" \
    --pruning-mode probability --seed 24680 --weights model \
    >"${ROOT}/logs/${RUN_NAME}_final_calibration_seed24680.log" 2>&1 &
pid1=$!
wait "${pid0}"
wait "${pid1}"
REPLAY_THRESHOLDS="${EVAL_ROOT}/onpolicy_step$(printf '%07d' "${REPLAY_FINAL_STEP}")_generated_iou_thresholds.json"
"${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning_replay.py \
    --replay-dirs "${CALIB_A}" "${CALIB_B}" \
    --ground-truth-dir "${TRAIN_LATENTS}" --output "${REPLAY_THRESHOLDS}" \
    --fine-resolution 32 --resolutions 8 16 32 \
    --objective iou --limit "${TRAIN_COUNT}"
REPLAY_F2_THRESHOLDS="${EVAL_ROOT}/onpolicy_step$(printf '%07d' "${REPLAY_FINAL_STEP}")_generated_f2_thresholds.json"
"${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning_replay.py \
    --replay-dirs "${CALIB_A}" "${CALIB_B}" \
    --ground-truth-dir "${TRAIN_LATENTS}" --output "${REPLAY_F2_THRESHOLDS}" \
    --fine-resolution 32 --resolutions 8 16 32 \
    --objective fbeta --fbeta 2 --limit "${TRAIN_COUNT}"
evaluate_two_seeds "onpolicy_clean_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")" \
    "${REPLAY_CHECKPOINT}" "${REPLAY_CLEAN_THRESHOLDS}"
evaluate_two_seeds "onpolicy_generated_iou_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")" \
    "${REPLAY_CHECKPOINT}" "${REPLAY_THRESHOLDS}" probability
evaluate_two_seeds "onpolicy_generated_f2_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")" \
    "${REPLAY_CHECKPOINT}" "${REPLAY_F2_THRESHOLDS}" probability

echo "===== Phase 7/7: paired held-out TRELLIS mesh evaluation (GPUs 0,1) ====="
for TAG in \
    "teacher_step$(printf '%07d' "${TEACHER_FINAL_STEP}")" \
    "onpolicy_clean_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")" \
    "onpolicy_generated_f2_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")"; do
    for SEED_GPU in "314159:0" "271828:1"; do
        SEED="${SEED_GPU%%:*}"
        GPU="${SEED_GPU##*:}"
        SAMPLES="${OUTPUT_ROOT}/samples_${TAG}_seed${SEED}"
        SPARSE_CONV_BACKEND=flex_gemm CUDA_VISIBLE_DEVICES="${GPU}" \
            "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
            --prediction-dir "${SAMPLES}" --ground-truth-dir "${HELDOUT_LATENTS}" \
            --output "${EVAL_ROOT}/${TAG}_seed${SEED}_mesh.json" \
            --limit "${MESH_LIMIT}" --resolution 512 --sample-points 50000 \
            >"${ROOT}/logs/${RUN_NAME}_${TAG}_mesh_seed${SEED}.log" 2>&1 &
    done
    wait
done

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_formal.py \
    --output-root "${OUTPUT_ROOT}" --eval-root "${EVAL_ROOT}" \
    --output-json "${EVAL_ROOT}/formal_summary.json" \
    --output-markdown "${EVAL_ROOT}/formal_summary.md"

echo "RDD3D_FORMAL_HELDOUT_COMPLETE"
echo "Split:       ${SPLIT_ROOT}/manifest.json"
echo "Zero-shot:   ${EVAL_ROOT}/zeroshot_step$(printf '%07d' "${INIT_STEP}")_seed314159_heldout_support.json"
echo "Teacher:     ${EVAL_ROOT}/teacher_step$(printf '%07d' "${TEACHER_FINAL_STEP}")_seed314159_heldout_support.json"
echo "On-policy balanced: ${EVAL_ROOT}/onpolicy_clean_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")_seed314159_heldout_support.json"
echo "On-policy speed:    ${EVAL_ROOT}/onpolicy_generated_iou_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")_seed314159_heldout_support.json"
echo "On-policy recall:   ${EVAL_ROOT}/onpolicy_generated_f2_cal_step$(printf '%07d' "${REPLAY_FINAL_STEP}")_seed314159_heldout_support.json"
