#!/usr/bin/env bash
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-260906}"
CKPT="${CKPT:-${ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
VAE_PATH="${VAE_PATH:-${ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/eval_results/transition_ablation/unified_l_300300400_400k}"
CONFIG="${OUT_ROOT}/paired_noise_filling_vs_innovation_preserving.jsonl"
LOG="${OUT_ROOT}/paired_noise_filling_vs_innovation_preserving.log"

test -f "${CKPT}"
test -d "${VAE_PATH}"
mkdir -p "${OUT_ROOT}"

# Filenames and output directories below are historical, not canonical labels.
# This historical comparison runs fresh versus legacy stage-initial-noise coupling.
# Current noise_filling instead recovers residual noise from the previous endpoint;
# it is a different variant and must not inherit these old results.
printf '%s\n' \
  "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-L/2\",\"ckpt\":\"${CKPT}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.0,\"upsample_mode\":\"fresh\",\"output_dir\":\"${OUT_ROOT}/noise_filling\"}" \
  "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-L/2\",\"ckpt\":\"${CKPT}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.0,\"upsample_mode\":\"innovation_preserving\",\"output_dir\":\"${OUT_ROOT}/innovation_preserving\"}" \
  > "${CONFIG}"

echo "GPU=${GPU}; samples/mode=${TOTAL_SAMPLES}; batch=${BATCH_SIZE}; seed=${SEED}"
echo "checkpoint=${CKPT}"
echo "outputs=${OUT_ROOT}"

cd "${ROOT}"
CUDA_VISIBLE_DEVICES="${GPU}" \
  "${ROOT}/.venv-rdd-train/bin/python" -u 2d/sample_fid_unified.py \
    --config "${CONFIG}" \
    --rank 0 \
    --world-size 1 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size "${BATCH_SIZE}" \
    --vae-batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --vae-path "${VAE_PATH}" \
    --no-eval \
  2>&1 | tee "${LOG}"

"${ROOT}/.venv-rdd-train/bin/python" 2d/make_transition_comparison_grid.py \
  --left "${OUT_ROOT}/noise_filling" \
  --right "${OUT_ROOT}/innovation_preserving" \
  --output "${OUT_ROOT}/paired_comparison.png"

echo "Completed paired transition samples: ${OUT_ROOT}"
