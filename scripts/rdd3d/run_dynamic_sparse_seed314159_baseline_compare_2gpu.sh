#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${ROOT}/.venv_rdd3d/bin/python"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
CONDITIONS="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/conditions_dinov2_vits14"
BASE_CKPT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/checkpoints/step0010000.pt"
BASE_THRESH="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_gate_w384d12_2gpu_thresholds.json"
FT_CKPT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/checkpoints/step0015000.pt"
FT_THRESH="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/step0015000_thresholds.json"
BASE_OUT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/samples_seed314159"
FT_OUT="${ROOT}/results_rdd3d/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/samples_step0015000_seed314159"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/strict_seed314159_compare"

mkdir -p "${BASE_OUT}" "${FT_OUT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${BASE_CKPT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${BASE_THRESH}" \
    --output-dir "${BASE_OUT}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit 4 --seed 314159 --weights model \
    >"${ROOT}/logs/rdd3d_seed314159_baseline10k.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${FT_CKPT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${FT_THRESH}" \
    --output-dir "${FT_OUT}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit 4 --seed 314159 --weights model \
    >"${ROOT}/logs/rdd3d_seed314159_replay15k.log" 2>&1 &
PID1=$!
wait "${PID0}"
wait "${PID1}"

"${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
    --prediction-dir "${BASE_OUT}" --ground-truth-dir "${LATENTS}" \
    --output "${EVAL_ROOT}/baseline10k.json" --resolution 32 \
    --stage-resolutions 8 16 32
"${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
    --prediction-dir "${FT_OUT}" --ground-truth-dir "${LATENTS}" \
    --output "${EVAL_ROOT}/replay15k.json" --resolution 32 \
    --stage-resolutions 8 16 32
echo "RDD3D_STRICT_SEED314159_COMPARE_COMPLETE"
