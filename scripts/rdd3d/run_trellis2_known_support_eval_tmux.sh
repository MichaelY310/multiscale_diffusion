#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ( "$1" != rdd && "$1" != flat ) ]]; then
    echo "Usage: $0 {rdd|flat}" >&2
    exit 2
fi
MODE="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRELLIS="${ROOT}/third_party/TRELLIS.2"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
RESOLUTION="${RDD3D_MODEL_RESOLUTION:-1024}"
RUN_NAME="${RDD3D_RUN_NAME:-pilot}"
DATA_ROOT="${RDD3D_EVAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
TRAIN_ROOT="${RDD3D_TRAIN_OUTPUT:-${ROOT}/results_rdd3d/trellis2_${MODE}_${RESOLUTION}_${RUN_NAME}}"
CONFIG="${RDD3D_EVAL_CONFIG:-${ROOT}/RDD_DiT_3D/configs/trellis2_${MODE}_${RESOLUTION}_${RUN_NAME}.json}"
GPU="${RDD3D_EVAL_GPU:-4}"
LIMIT="${RDD3D_EVAL_LIMIT:-1}"
SUPPORT_ARGS=""
if [[ -n "${RDD3D_EVAL_SUPPORTS:-}" ]]; then
    SUPPORT_ARGS="--supports \"${RDD3D_EVAL_SUPPORTS}\""
fi
EXTRA_ARGS=""
if [[ "${RDD3D_EVAL_SKIP_DECODE:-0}" == 1 ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --skip-decode"
fi
if [[ "${RDD3D_EVAL_SAVE_LATENTS:-0}" == 1 ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --save-latents"
fi
if [[ "${RDD3D_EVAL_SAVE_MESHES:-0}" == 1 ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --save-meshes"
fi

CHECKPOINT="${RDD3D_EVAL_CHECKPOINT:-}"
if [[ -z "${CHECKPOINT}" ]]; then
    CHECKPOINT="$(find "${TRAIN_ROOT}/ckpts" -maxdepth 1 -type f -name 'denoiser_ema*_step*.pt' 2>/dev/null | sort | tail -1)"
fi
if [[ -z "${CHECKPOINT}" || ! -f "${CHECKPOINT}" ]]; then
    echo "No evaluation checkpoint found. Set RDD3D_EVAL_CHECKPOINT." >&2
    exit 2
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Missing training config: ${CONFIG}" >&2
    exit 2
fi

LATENTS="${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${RESOLUTION}"
OUTPUT="${ROOT}/eval_results/rdd3d_known_support/${MODE}_${RESOLUTION}_${RUN_NAME}.json"
LOG="${ROOT}/logs/rdd3d_known_support_${MODE}_${RESOLUTION}_${RUN_NAME}.log"
SESSION="rdd3d_eval_${MODE}_${RESOLUTION}_${RUN_NAME}"
mkdir -p "$(dirname "${OUTPUT}")" "${ROOT}/logs"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && CUDA_VISIBLE_DEVICES=\"${GPU}\" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPARSE_CONV_BACKEND=flex_gemm ATTN_BACKEND=xformers PYTHONPATH=\"${TRELLIS}:${ROOT}/RDD_DiT_3D\" \"${PYTHON}\" RDD_DiT_3D/evaluate_trellis2_known_support.py --mode \"${MODE}\" --config \"${CONFIG}\" --checkpoint \"${CHECKPOINT}\" --latents \"${LATENTS}\" --renders \"${DATA_ROOT}/renders_cond\" ${SUPPORT_ARGS} ${EXTRA_ARGS} --output \"${OUTPUT}\" --limit \"${LIMIT}\" 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION} on GPU ${GPU}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Attach: tmux attach -t ${SESSION}"
echo "Log:    tail -f ${LOG}"
