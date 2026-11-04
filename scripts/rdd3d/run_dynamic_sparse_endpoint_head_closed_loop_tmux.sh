#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_RUN="${RDD3D_FORMAL_BASE_RUN_NAME:-dynamic_sparse_formal_96train_32heldout_w384d12}"
HEAD_RUN="${BASE_RUN}_endpoint_head"
HEAD_OUTPUT="${ROOT}/results_rdd3d/${HEAD_RUN}"
HEAD_EVAL="${ROOT}/eval_results/rdd3d_dynamic_sparse/${HEAD_RUN}"
SESSION="${RDD3D_ENDPOINT_CLOSED_LOOP_SESSION:-rdd3d_endpoint_head_closed_loop}"
LOG="${ROOT}/logs/rdd3d_endpoint_head_closed_loop.log"

mkdir -p "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

COMMAND="cd '${ROOT}' && \
RDD3D_FORMAL_RUN_NAME='${HEAD_RUN}' \
RDD3D_CLOSED_LOOP_CHECKPOINT='${HEAD_OUTPUT}/endpoint_head/checkpoint.pt' \
RDD3D_CLOSED_LOOP_CLEAN_THRESHOLDS='${HEAD_EVAL}/endpoint_head_clean_thresholds.json' \
RDD3D_CLOSED_LOOP_INITIAL_IOU='${HEAD_EVAL}/endpoint_head_generated_iou_thresholds.json' \
RDD3D_CLOSED_LOOP_INITIAL_F2='${HEAD_EVAL}/endpoint_head_generated_f2_thresholds.json' \
RDD3D_CLOSED_LOOP_OUTPUT_STEM='endpoint_head_closed_loop' \
RDD3D_CLOSED_LOOP_TAG_PREFIX='endpoint_head_closed_loop' \
RDD3D_CLOSED_LOOP_TAG_SUFFIX='' \
bash scripts/rdd3d/run_dynamic_sparse_closed_loop_calibration_2gpu.sh 2>&1 | tee '${LOG}'"
tmux new-session -d -s "${SESSION}" "${COMMAND}"
echo "Started ${SESSION}"
echo "Log: ${LOG}"
