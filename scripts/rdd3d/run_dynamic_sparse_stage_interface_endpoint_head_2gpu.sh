#!/usr/bin/env bash
set -euo pipefail

# Freeze both generative models and fit only the 385-parameter occupancy head
# of the 34K refinement model on endpoint states from the final upstream policy.
# Stage 8 remains the untouched 19K model; only stages 16/32 receive head
# updates.  The fitted model is then evaluated with a fresh strict F2 loop.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
DATA_ROOT="${RDD3D_FORMAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_github_train_256}"
SPLIT_ROOT="${RDD3D_FORMAL_SPLIT_ROOT:-${DATA_ROOT}/rdd_dynamic_split_96_32_seed98317}"
TRAIN_LATENTS="${SPLIT_ROOT}/train_latents"
CONDITIONS="${SPLIT_ROOT}/conditions_dinov2_vits14"
RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

COARSE_CHECKPOINT="${RDD3D_INTERFACE_COARSE_CHECKPOINT:-${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu/checkpoints/step0019000.pt}"
REFINE_CHECKPOINT="${RDD3D_INTERFACE_FINAL_CHECKPOINT:-${OUTPUT_ROOT}/stage_interface_19coarse_refine/refine_adapt/checkpoints/step0034000.pt}"
COARSE_THRESHOLDS="${RDD3D_INTERFACE_COARSE_THRESHOLDS:-${EVAL_ROOT}/zeroshot_step0019000_thresholds.json}"
REPLAY_ROOT="${OUTPUT_ROOT}/stage_interface_19coarse_refine/closed_loop_f2"
REPLAY_A="${REPLAY_ROOT}/f2_after16_seed32601"
REPLAY_B="${REPLAY_ROOT}/f2_after16_seed32602"

HEAD_ROOT="${OUTPUT_ROOT}/stage_interface_19coarse_refine/endpoint_head"
HEAD_CHECKPOINT="${HEAD_ROOT}/checkpoint.pt"
HEAD_REPORT="${EVAL_ROOT}/interface19coarse_refine34k_endpoint_head_training.json"
HEAD_CLEAN_RAW="${EVAL_ROOT}/interface19coarse_refine34k_endpoint_head_clean_raw.json"
HEAD_CLEAN_THRESHOLDS="${EVAL_ROOT}/interface19coarse_refine34k_endpoint_head_clean_thresholds.json"

for REQUIRED in "${PYTHON}" "${TRAIN_LATENTS}" "${CONDITIONS}" \
    "${COARSE_CHECKPOINT}" "${REFINE_CHECKPOINT}" "${COARSE_THRESHOLDS}" \
    "${REPLAY_A}/manifest.json" "${REPLAY_B}/manifest.json"; do
    if [[ ! -e "${REQUIRED}" ]]; then
        echo "Missing stage-interface endpoint-head input: ${REQUIRED}" >&2
        exit 2
    fi
done
mkdir -p "${HEAD_ROOT}" "${EVAL_ROOT}" "${ROOT}/logs"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

echo "===== Interface endpoint head 1/3: frozen-denoiser fitting (GPU 0) ====="
if [[ ! -s "${HEAD_CHECKPOINT}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" \
        RDD_DiT_3D/finetune_dynamic_sparse_occupancy_replay.py \
        --checkpoint "${REFINE_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --replay-dirs "${REPLAY_A}" "${REPLAY_B}" \
        --output "${HEAD_CHECKPOINT}" --report "${HEAD_REPORT}" \
        --weights model --steps 2000 --lr 5e-4 --positive-weight 3 \
        --stage-probabilities 0 0.40 0.60 --seed 20260813 --amp bf16 \
        >"${ROOT}/logs/${RUN_NAME}_interface_endpoint_head_fit.log" 2>&1
fi

echo "===== Interface endpoint head 2/3: clean threshold composition ====="
if [[ ! -s "${HEAD_CLEAN_RAW}" ]]; then
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${HEAD_CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${HEAD_CLEAN_RAW}" \
        --target-recall 0.999 0.995 0.99 --limit 96 --weights model
fi
if [[ ! -s "${HEAD_CLEAN_THRESHOLDS}" ]]; then
    "${PYTHON}" RDD_DiT_3D/compose_dynamic_sparse_thresholds.py \
        --base "${HEAD_CLEAN_RAW}" --generated "${COARSE_THRESHOLDS}" \
        --replace-resolutions 8 --output "${HEAD_CLEAN_THRESHOLDS}"
fi

echo "===== Interface endpoint head 3/3: fresh strict sequential F2 ====="
export RDD3D_CLOSED_LOOP_CHECKPOINT="${HEAD_CHECKPOINT}"
export RDD3D_CLOSED_LOOP_CLEAN_THRESHOLDS="${HEAD_CLEAN_THRESHOLDS}"
export RDD3D_CLOSED_LOOP_INITIAL_F2="${EVAL_ROOT}/interface19coarse_refine34k_endpoint_head_generated_f2_thresholds.json"
export RDD3D_CLOSED_LOOP_OUTPUT_STEM="interface19coarse_refine34k_endpoint_head_closed_loop"
export RDD3D_CLOSED_LOOP_TAG_PREFIX="interface19coarse_refine34k_endpoint_head_closed_loop"
export RDD3D_CLOSED_LOOP_TAG_SUFFIX=""
export RDD3D_CLOSED_LOOP_LABELS="f2"
export RDD3D_CLOSED_LOOP_WORK="${HEAD_ROOT}/closed_loop_f2"
export RDD3D_CLOSED_LOOP_STAGE_CHECKPOINTS="${COARSE_CHECKPOINT} ${HEAD_CHECKPOINT} ${HEAD_CHECKPOINT}"
export RDD3D_CLOSED_LOOP_STAGE_WEIGHTS="model model model"
bash "${ROOT}/scripts/rdd3d/run_dynamic_sparse_closed_loop_calibration_2gpu.sh"

echo "RDD3D_STAGE_INTERFACE_ENDPOINT_HEAD_COMPLETE"
echo "Head audit: ${HEAD_REPORT}"
