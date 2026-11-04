#!/usr/bin/env bash
set -euo pipefail

# Strict generated-policy F2 calibration for the stage-interface model:
# 19K checkpoint at 8^3, interface-adapted 34K checkpoint at 16^3/32^3.
# The generic driver first creates a one-shot generated calibration when it is
# absent, then re-rolls after fixing 8 and 16 so every threshold is calibrated
# on the candidate distribution induced by its upstream pruning policy.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RUN_NAME="${RDD3D_FORMAL_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
OUTPUT_ROOT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

COARSE_CHECKPOINT="${RDD3D_INTERFACE_COARSE_CHECKPOINT:-${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu/checkpoints/step0019000.pt}"
REFINE_CHECKPOINT="${RDD3D_INTERFACE_FINAL_CHECKPOINT:-${OUTPUT_ROOT}/stage_interface_19coarse_refine/refine_adapt/checkpoints/step0034000.pt}"
CLEAN_THRESHOLDS="${RDD3D_INTERFACE_FINAL_THRESHOLDS:-${EVAL_ROOT}/interface19coarse_refine34k_clean_thresholds.json}"

export RDD3D_CLOSED_LOOP_CHECKPOINT="${REFINE_CHECKPOINT}"
export RDD3D_CLOSED_LOOP_CLEAN_THRESHOLDS="${CLEAN_THRESHOLDS}"
export RDD3D_CLOSED_LOOP_INITIAL_F2="${EVAL_ROOT}/interface19coarse_refine34k_generated_f2_thresholds.json"
export RDD3D_CLOSED_LOOP_OUTPUT_STEM="interface19coarse_refine34k_closed_loop"
export RDD3D_CLOSED_LOOP_TAG_PREFIX="interface19coarse_refine34k_closed_loop"
export RDD3D_CLOSED_LOOP_TAG_SUFFIX=""
export RDD3D_CLOSED_LOOP_LABELS="f2"
export RDD3D_CLOSED_LOOP_WORK="${OUTPUT_ROOT}/stage_interface_19coarse_refine/closed_loop_f2"
export RDD3D_CLOSED_LOOP_STAGE_CHECKPOINTS="${COARSE_CHECKPOINT} ${REFINE_CHECKPOINT} ${REFINE_CHECKPOINT}"
export RDD3D_CLOSED_LOOP_STAGE_WEIGHTS="model model model"

exec bash "${ROOT}/scripts/rdd3d/run_dynamic_sparse_closed_loop_calibration_2gpu.sh"
