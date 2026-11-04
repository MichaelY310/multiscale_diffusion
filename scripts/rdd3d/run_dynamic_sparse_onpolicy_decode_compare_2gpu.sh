#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
RUN="${ROOT}/results_rdd3d/dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/decode_compare_seed314159"
mkdir -p "${ROOT}/logs" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/third_party/TRELLIS.2:${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"
export SPARSE_CONV_BACKEND=flex_gemm

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
    --prediction-dir "${RUN}/samples_step0019000_seed314159" \
    --ground-truth-dir "${LATENTS}" --output "${EVAL_ROOT}/onpolicy19k.json" \
    --limit 4 --resolution 512 --sample-points 50000 \
    >"${ROOT}/logs/rdd3d_decode_onpolicy19k.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
    --prediction-dir "${RUN}/samples_step0020000_seed314159" \
    --ground-truth-dir "${LATENTS}" --output "${EVAL_ROOT}/onpolicy20k.json" \
    --limit 4 --resolution 512 --sample-points 50000 \
    >"${ROOT}/logs/rdd3d_decode_onpolicy20k.log" 2>&1 &
PID1=$!
STATUS=0
wait "${PID0}" || STATUS=$?
wait "${PID1}" || STATUS=$?
if (( STATUS != 0 )); then
    tail -n 60 "${ROOT}/logs/rdd3d_decode_onpolicy19k.log" || true
    tail -n 60 "${ROOT}/logs/rdd3d_decode_onpolicy20k.log" || true
    exit "${STATUS}"
fi
cat "${EVAL_ROOT}/onpolicy19k.json"
cat "${EVAL_ROOT}/onpolicy20k.json"
echo "RDD3D_ONPOLICY_DECODE_COMPARE_COMPLETE"
