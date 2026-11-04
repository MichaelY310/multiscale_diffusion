#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRELLIS="${ROOT}/third_party/TRELLIS.2"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
RESOLUTION="${RDD3D_MODEL_RESOLUTION:-1024}"
DATA_ROOT="${RDD3D_EVAL_DATA_ROOT:-${ROOT}/datasets/rdd3d_formal/ObjaverseXL_sketchfab_validation_1000}"
LATENTS="${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${RESOLUTION}"
OUTPUT="${RDD3D_SUPPORT_OUTPUT:-${ROOT}/eval_results/rdd3d_predicted_supports/${RESOLUTION}}"
LIMIT="${RDD3D_EVAL_LIMIT:-100}"
GPU="${RDD3D_EVAL_GPU:-4}"
SESSION="rdd3d_supports_${RESOLUTION}"
LOG="${ROOT}/logs/${SESSION}.log"

mkdir -p "${ROOT}/logs" "${OUTPUT}"
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi

tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && CUDA_VISIBLE_DEVICES=\"${GPU}\" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPARSE_CONV_BACKEND=flex_gemm ATTN_BACKEND=xformers PYTHONPATH=\"${TRELLIS}:${ROOT}/RDD_DiT_3D\" \"${PYTHON}\" RDD_DiT_3D/generate_trellis2_predicted_supports.py --latents \"${LATENTS}\" --renders \"${DATA_ROOT}/renders_cond\" --output \"${OUTPUT}\" --resolution \"${RESOLUTION}\" --limit \"${LIMIT}\" 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION} on GPU ${GPU}"
echo "Shared supports: ${OUTPUT}"
echo "Log: tail -f ${LOG}"
