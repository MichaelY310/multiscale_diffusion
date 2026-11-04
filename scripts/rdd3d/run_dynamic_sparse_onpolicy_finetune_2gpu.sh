#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${RDD3D_DYNAMIC_PYTHON:-${ROOT}/.venv_rdd3d/bin/python}"
LATENTS="${ROOT}/datasets/rdd3d_pilot/ObjaverseXL_github/shape_latents/shape_enc_next_dc_f16c32_fp16_512"
CONDITIONS="${ROOT}/results_rdd3d/dynamic_sparse_32asset_gate_w384d12_2gpu/conditions_dinov2_vits14"
BASE_REPLAY="${ROOT}/results_rdd3d/dynamic_sparse_10k_generated_parent_replay"
PARENT_RUN="${ROOT}/results_rdd3d/dynamic_sparse_32asset_replay_ft_w384d12_2gpu"
PARENT_CHECKPOINT="${PARENT_RUN}/checkpoints/step0015000.pt"
PARENT_THRESHOLDS="${ROOT}/eval_results/rdd3d_dynamic_sparse/dynamic_sparse_32asset_replay_ft_w384d12_2gpu/step0015000_thresholds.json"
REPLAY_DEFAULT="${ROOT}/results_rdd3d/dynamic_sparse_15k_replay_default_seed424242"
REPLAY_PROB="${ROOT}/results_rdd3d/dynamic_sparse_15k_replay_probability_seed777777"
RUN_NAME="dynamic_sparse_32asset_onpolicy_ft_w384d12_2gpu"
OUTPUT="${ROOT}/results_rdd3d/${RUN_NAME}"
EVAL_ROOT="${ROOT}/eval_results/rdd3d_dynamic_sparse/${RUN_NAME}"

mkdir -p "${REPLAY_DEFAULT}" "${REPLAY_PROB}" "${OUTPUT}" "${EVAL_ROOT}"
cd "${ROOT}"
export PYTHONPATH="${ROOT}/RDD_DiT_3D:${PYTHONPATH:-}"

echo "===== Phase 1/3: two on-policy replay trajectories (GPUs 0,1) ====="
CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${PARENT_CHECKPOINT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${PARENT_THRESHOLDS}" \
    --output-dir "${REPLAY_DEFAULT}" --steps 8 12 20 \
    --mass-thresholds 0.05 0.10 0.25 --limit 32 --seed 424242 \
    --weights model >"${ROOT}/logs/rdd3d_onpolicy_replay_default.log" 2>&1 &
PID0=$!
CUDA_VISIBLE_DEVICES=1 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
    --checkpoint "${PARENT_CHECKPOINT}" --latent-dir "${LATENTS}" \
    --condition-dir "${CONDITIONS}" --thresholds "${PARENT_THRESHOLDS}" \
    --output-dir "${REPLAY_PROB}" --steps 8 12 20 \
    --probability-thresholds 0.15 0.60 0.90 \
    --mass-thresholds 0.05 0.10 0.25 --pruning-mode probability \
    --limit 32 --seed 777777 --weights model \
    >"${ROOT}/logs/rdd3d_onpolicy_replay_probability.log" 2>&1 &
PID1=$!
wait "${PID0}"
wait "${PID1}"

echo "===== Phase 2/3: mixed on-policy RDD fine-tuning (GPUs 0,1) ====="
CUDA_VISIBLE_DEVICES=0,1 "${PYTHON}" -m torch.distributed.run \
    --standalone --nproc_per_node 2 RDD_DiT_3D/train_dynamic_sparse_rdd.py \
    --latent-dir "${LATENTS}" --condition-dir "${CONDITIONS}" \
    --output "${OUTPUT}" --resume "${PARENT_CHECKPOINT}" \
    --resume-weights model --reset-optimizer \
    --fine-resolution 32 --resolutions 8 16 32 \
    --time-fractions 0.5 0.3 0.2 --max-assets 32 \
    --steps 5000 --width 384 --depth 12 --lr 2e-5 \
    --stage-probabilities 0.25 0.30 0.45 \
    --generated-parent-dirs "${BASE_REPLAY}" "${REPLAY_DEFAULT}" "${REPLAY_PROB}" \
    --generated-parent-probability 0.75 \
    --generated-endpoint-time-probability 0.50 \
    --generated-parent-velocity-weight 0.75 \
    --ema-rate 0.999 --occupancy-loss-weight 0.75 --positive-weight 3.0 \
    --negative-parent-ratio 1.0 --endpoint-time-probability 0.25 \
    --mass-velocity-weight 4 --save-every 1000 --log-every 25 --amp bf16

echo "===== Phase 3/3: novel-seed checkpoint evaluation (GPU 0) ====="
for STEP in 16000 17000 18000 19000 20000; do
    TAG="step$(printf '%07d' "${STEP}")"
    CHECKPOINT="${OUTPUT}/checkpoints/${TAG}.pt"
    THRESHOLDS="${EVAL_ROOT}/${TAG}_thresholds.json"
    SAMPLES="${OUTPUT}/samples_${TAG}_seed314159"
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/calibrate_dynamic_sparse_pruning.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --output "${THRESHOLDS}" \
        --target-recall 0.999 0.995 0.99 --limit 32 --weights model
    CUDA_VISIBLE_DEVICES=0 "${PYTHON}" RDD_DiT_3D/sample_dynamic_sparse_dataset.py \
        --checkpoint "${CHECKPOINT}" --latent-dir "${LATENTS}" \
        --condition-dir "${CONDITIONS}" --thresholds "${THRESHOLDS}" \
        --output-dir "${SAMPLES}" --steps 8 12 20 \
        --mass-thresholds 0.05 0.10 0.25 --limit 4 \
        --seed 314159 --weights model
    "${PYTHON}" RDD_DiT_3D/evaluate_dynamic_sparse_predictions.py \
        --prediction-dir "${SAMPLES}" --ground-truth-dir "${LATENTS}" \
        --output "${EVAL_ROOT}/${TAG}_seed314159_generated_support.json" \
        --resolution 32 --stage-resolutions 8 16 32
done

"${PYTHON}" RDD_DiT_3D/summarize_dynamic_sparse_finetune.py \
    --run-dir "${OUTPUT}" --eval-dir "${EVAL_ROOT}" --seed 314159 \
    --output "${EVAL_ROOT}/summary_seed314159.json"

echo "RDD3D_ONPOLICY_FINETUNE_COMPLETE"
