#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
BASE_RUN="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu"
BASE_CHECKPOINT="${BASE_RUN}/checkpoints/step0010000.pt"
CONDITIONS="${BASE_RUN}/conditions_dinov2_vits14"
BASE_THRESHOLDS="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_gate_w384d12_2gpu_thresholds.json"
REPLAY="${ROOT}/results_rdd3d/dynamic_sparse_10k_generated_parent_replay"
RUN_NAME="dynamic_sparse_32asset_replay_ft_w384d12_2gpu"
OUTPUT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

mkdir -p "${REPLAY}" "${OUTPUT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

echo "===== Phase 1/3: generate inference-distribution parent replay (GPU 0) ====="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${BASE_CHECKPOINT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${BASE_THRESHOLDS}" \
    --output-dir "${REPLAY}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit 32 --weights model

echo "===== Phase 2/3: replay-aware RDD fine-tuning (GPUs 0,1) ====="
CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node 2 RDD_DiT_3D/train_dynamic_sparse_rdd.py \
    --latent-dir "${LATENTS}" --condition-dir "${CONDITIONS}" \
    --output "${OUTPUT}" --resume "${BASE_CHECKPOINT}" \
    --resume-weights model --reset-optimizer \
    --fine-resolution 32 --resolutions 8 16 32 \
    --time-fractions 0.5 0.3 0.2 --max-assets 32 \
    --steps 5000 --width 384 --depth 12 --lr 5e-5 \
    --stage-probabilities 0.15 0.30 0.55 \
    --generated-parent-dir "${REPLAY}" \
    --generated-parent-probability 0.60 \
    --generated-endpoint-time-probability 0.70 \
    --generated-parent-velocity-weight 0.10 \
    --ema-rate 0.999 --occupancy-loss-weight 1.0 --positive-weight 3.0 \
    --negative-parent-ratio 1.0 --endpoint-time-probability 0.25 \
    --mass-velocity-weight 4 --save-every 1000 --log-every 25 --amp bf16

echo "===== Phase 3/3: evaluate every 1K continuation checkpoint (GPU 0) ====="
for STEP in 11000 12000 13000 14000 15000; do
    TAG="step$(printf '%07d' "${STEP}")"
    CHECKPOINT="${OUTPUT}/checkpoints/${TAG}.pt"
    THRESHOLDS="${EVAL_ROOT}/${TAG}_thresholds.json"
    SAMPLES="${OUTPUT}/samples_${TAG}"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit 32 --weights model
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${SAMPLES}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --limit 4 --weights model
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${SAMPLES}" --ground-truth-dir "${LATENTS}" \
        --output "${EVAL_ROOT}/${TAG}_generated_support.json" \
        --resolution 32 --stage-resolutions 8 16 32
done

echo "RDD3D_REPLAY_FINETUNE_COMPLETE"
