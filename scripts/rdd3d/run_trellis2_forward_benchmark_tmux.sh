#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DATA_ROOT="${RDD3D_DATA_ROOT:-${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github}"
GPU="${RDD3D_GPU:-4}"
MODEL_RESOLUTION="${RDD3D_MODEL_RESOLUTION:-512}"
SESSION="rdd3d_forward_benchmark_${MODEL_RESOLUTION}"
LOG="${ROOT}/logs/rdd3d_forward_benchmark_${MODEL_RESOLUTION}.log"
STEM="slat_flow_img2shape_dit_1_3B_${MODEL_RESOLUTION}_bf16"
LATENTS="${DATA_ROOT}/shape_latents/shape_enc_next_dc_f16c32_fp16_${MODEL_RESOLUTION}"
if [[ "${MODEL_RESOLUTION}" == 1024 ]]; then
    CONFIG="${ROOT}/third_party/TRELLIS.2/configs/gen/slat_flow_img2shape_dit_1_3B_512_bf16_ft1024.json"
else
    CONFIG="${ROOT}/third_party/TRELLIS.2/configs/gen/${STEM}.json"
fi
CHECKPOINT="${ROOT}/checkpoints/pretrained/trellis2/${STEM}.pt"
OUTPUT="${ROOT}/eval_results/rdd3d_pilot/forward_benchmark_${MODEL_RESOLUTION}.json"

if [[ ! -f "${CHECKPOINT}" ]]; then
    "${ROOT}/.venv_rdd3d/bin/python" "${ROOT}/scripts/rdd3d/download_trellis2_rdd_checkpoints.py" \
        --output-dir "${ROOT}/checkpoints/pretrained/trellis2" --resolution "${MODEL_RESOLUTION}"
fi

if tmux has-session -t "${SESSION}" 2>/dev/null; then
    echo "tmux session already exists: ${SESSION}" >&2
    exit 2
fi
mkdir -p "${ROOT}/logs" "$(dirname "${OUTPUT}")"
tmux new-session -d -s "${SESSION}" \
    "bash -lc 'cd \"${ROOT}\" && CUDA_VISIBLE_DEVICES=\"${GPU}\" SPARSE_CONV_BACKEND=flex_gemm ATTN_BACKEND=xformers TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONPATH=\"${ROOT}/third_party/TRELLIS.2:${ROOT}/RDD_DiT_3D\" \"${ROOT}/.venv_rdd3d/bin/python\" RDD_DiT_3D/benchmark_trellis2_rdd_forward.py \"${LATENTS}\" --config \"${CONFIG}\" --checkpoint \"${CHECKPOINT}\" --output \"${OUTPUT}\" --num-assets \"${RDD3D_BENCH_ASSETS:-4}\" --warmup \"${RDD3D_BENCH_WARMUP:-2}\" --iterations \"${RDD3D_BENCH_ITERS:-5}\" 2>&1 | tee \"${LOG}\"'"

echo "Started ${SESSION}; log: ${LOG}"
