#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
SPLIT="${ROOT}/datasets/rdd3d_scale1k/ObjaverseXL_github_train_1280/rdd_dynamic_split_896train_96dev_96test_seed104729"
TRAIN_LATENTS="${SPLIT}/train_latents"
DEV_LATENTS="${SPLIT}/dev_latents"
CONDITIONS="${SPLIT}/conditions_dinov2_vits14"
RENDERS="${ROOT}/datasets/rdd3d_scale1k/ObjaverseXL_github_train_1280/renders_cond"
RUN="${ROOT}/results_rdd3d/dynamic_sparse_scale1k_w640d16_4gpu"
CHECKPOINT="${RUN}/onpolicy/checkpoints/step0100000.pt"
EVAL="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_scale1k_w640d16_4gpu"
THRESHOLDS="${EVAL}/onpolicy_step100000_preview_thresholds.json"
SAMPLES="${RUN}/preview_step100000/samples"
MESHES="${RUN}/preview_step100000/meshes"
LIMIT="${RDD3D_PREVIEW_LIMIT:-4}"
CALIBRATION_LIMIT="${RDD3D_PREVIEW_CALIBRATION_LIMIT:-128}"

cd "${ROOT}"
mkdir -p "${SAMPLES}" "${MESHES}" "${EVAL}" "${ROOT}/logs"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${ROOT}/third_party/TRELLIS.2:${PYTHONPATH:-}"

if [[ ! -s "${THRESHOLDS}" ]]; then
    echo "===== 1/3: calibrate 100K EMA pruning thresholds on train-only assets ====="
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${TRAIN_LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit "${CALIBRATION_LIMIT}" \
        --weights ema
fi

echo "===== 2/3: sample ${LIMIT} unseen dev assets with dynamic pruning ====="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${CHECKPOINT}" --latent-dir "${DEV_LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
    --output-dir "${SAMPLES}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --pruning-mode or \
    --limit "${LIMIT}" --seed 20260814 --weights ema

echo "===== 3/3: decode TRELLIS meshes and render four-view previews ====="
SPARSE_CONV_BACKEND=flex_gemm CUDA_VISIBLE_DEVICES=1 "${PYTHON}" \
    RDD_DiT_3D/decode_render_dynamic_sparse_samples.py \
    --prediction-dir "${SAMPLES}" --render-dir "${RENDERS}" \
    --output-dir "${MESHES}" --limit "${LIMIT}" --resolution 512 \
    --preview-resolution 512

echo "RDD3D_SCALE1K_PREVIEWS_COMPLETE"
echo "Preview manifest: ${MESHES}/manifest.json"
