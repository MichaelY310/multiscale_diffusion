#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
TRELLIS="${ROOT}/third_party/TRELLIS.2"
DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
RESOLUTION="${RDD3D_OVOXEL_RESOLUTION:-512}"
LATENT_RESOLUTION=$((RESOLUTION / 16))
LATENTS="${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${RESOLUTION}"
REPORT="${ROOT}/eval_results/rdd3d_pilot/slat_decode_pyramid_${RESOLUTION}.json"
LOG="${ROOT}/logs/rdd3d_slat_decode_pilot_${RESOLUTION}.log"
SESSION="rdd3d_slat_decode_${RESOLUTION}"
GPU="${RDD3D_GPU:-4}"

if [[ ! -d "${LATENTS}" ]]; then
    echo "Missing ${LATENTS}; finish the representation pilot first." >&2
    exit 2
fi
if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi
mkdir -p "$(dirname "${REPORT}")" "${ROOT}/logs"

tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && CUDA_VISIBLE_DEVICES=\"${GPU}\" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True SPARSE_CONV_BACKEND=flex_gemm PYTHONPATH=\"${TRELLIS}:${ROOT}/RDD_DiT_3D\" \"${PYTHON}\" RDD_DiT_3D/evaluate_slat_pyramid_decode.py \"${LATENTS}\" --output \"${REPORT}\" --resolution \"${RESOLUTION}\" --latent-resolution \"${LATENT_RESOLUTION}\" --limit \"${RDD3D_DECODE_LIMIT:-8}\" 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION}; log: ${LOG}"
