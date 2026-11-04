#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-4}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-16}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
DMD_CHECKPOINT="${DMD_CHECKPOINT:-${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_rdd_dmd_full_v2_gb256/checkpoints/latest.pt}"
DMD_OUTPUT="${DMD_OUTPUT:-${REPO_ROOT}/eval_results/previews/rdd_dmd_full_v2_step000552_3nfe}"
TWO_STAGE_CONFIG="${TWO_STAGE_CONFIG:-${REPO_ROOT}/2d/exp/preview_unified_l_2stage_500500_step285000.jsonl}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval_results/previews/logs}"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
mkdir -p "${DMD_OUTPUT}" "${LOG_DIR}"

echo "[1/2] Sampling DMD v2 checkpoint with exactly three model evaluations"
CUDA_VISIBLE_DEVICES="${GPU}" python -u 2d/sample_rdd_phased_dmd3.py \
    --checkpoint "${DMD_CHECKPOINT}" \
    --weight-source ema \
    --output-dir "${DMD_OUTPUT}" \
    --vae-path "${VAE_PATH}" \
    --model DiT-L/2 \
    --resolutions 32,16,8 \
    --stage-lengths 300,300,400 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size 8 \
    --vae-batch-size 8 \
    --rank 0 --world-size 1 --seed 0 --bf16 \
    2>&1 | tee "${LOG_DIR}/rdd_dmd_full_v2_step000552_3nfe.log"

echo "[2/2] Sampling RDD Unified-L 2-stage 500/500 at step 285K"
CUDA_VISIBLE_DEVICES="${GPU}" python -u 2d/sample_fid_unified.py \
    --config "${TWO_STAGE_CONFIG}" \
    --rank 0 --world-size 1 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size 8 \
    --vae-batch-size 8 \
    --seed 0 \
    --vae-path "${VAE_PATH}" \
    --no-eval \
    2>&1 | tee "${LOG_DIR}/rdd_unified_l_2stage_500500_step285000_ddpm250_cfg1.5.log"

dmd_count="$(find "${DMD_OUTPUT}" -maxdepth 1 -type f -name '*.png' | wc -l)"
two_stage_output="${REPO_ROOT}/eval_results/previews/rdd_unified_l_2stage_500500_step285000_ddpm250_cfg1.5"
two_stage_count="$(find "${two_stage_output}" -maxdepth 1 -type f -name '*.png' | wc -l)"
echo "DMD preview images: ${dmd_count}/${TOTAL_SAMPLES} in ${DMD_OUTPUT}"
echo "Two-stage preview images: ${two_stage_count}/${TOTAL_SAMPLES} in ${two_stage_output}"
[[ "${dmd_count}" -eq "${TOTAL_SAMPLES}" ]]
[[ "${two_stage_count}" -eq "${TOTAL_SAMPLES}" ]]
