#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
BASELINE="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/samples_seed20260811"
REPLAY15K="${ROOT}/results_rdd3d/dynamic_sparse_15k_threshold_ablations_seed20260811/probability_only_relaxed"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/decode_compare_seed20260811"

mkdir -p "${ROOT}/logs" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/third_party/TRELLIS.2:${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"
export SPARSE_CONV_BACKEND=flex_gemm

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
    --prediction-dir "${REPLAY15K}" --ground-truth-dir "${LATENTS}" \
    --output "${EVAL_ROOT}/replay15k_probability_only_relaxed.json" \
    --limit 4 --resolution 512 --sample-points 50000 \
    >"${ROOT}/logs/rdd3d_decode_replay15k.log" 2>&1 &
PID0=$!

CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_decode.py \
    --prediction-dir "${BASELINE}" --ground-truth-dir "${LATENTS}" \
    --output "${EVAL_ROOT}/baseline10k.json" \
    --limit 4 --resolution 512 --sample-points 50000 \
    >"${ROOT}/logs/rdd3d_decode_baseline10k.log" 2>&1 &
PID1=$!

STATUS=0
wait "${PID0}" || STATUS=$?
wait "${PID1}" || STATUS=$?
if (( STATUS != 0 )); then
    tail -n 60 "${ROOT}/logs/rdd3d_decode_replay15k.log" || true
    tail -n 60 "${ROOT}/logs/rdd3d_decode_baseline10k.log" || true
    exit "${STATUS}"
fi

cat "${EVAL_ROOT}/baseline10k.json"
cat "${EVAL_ROOT}/replay15k_probability_only_relaxed.json"
echo "RDD3D_DECODE_COMPARE_COMPLETE"
