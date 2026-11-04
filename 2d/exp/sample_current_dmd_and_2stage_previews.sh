#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
DMD_CHECKPOINT="${DMD_CHECKPOINT:-${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_phased_dmd3_dmd_imagenet_gb256/checkpoints/latest.pt}"
DMD_OUTPUT="${DMD_OUTPUT:-${REPO_ROOT}/eval_results/previews/rdd_unified_l_phased_dmd3_latest_3nfe}"
TWO_STAGE_CONFIG="${TWO_STAGE_CONFIG:-${REPO_ROOT}/2d/exp/preview_unified_l_2stage_500500_step097169.jsonl}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/eval_results/previews/logs}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-16}"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
mkdir -p "${DMD_OUTPUT}" "${LOG_DIR}"

CUDA_VISIBLE_DEVICES=4 python -u 2d/sample_rdd_phased_dmd3.py \
    --checkpoint "${DMD_CHECKPOINT}" \
    --output-dir "${DMD_OUTPUT}" \
    --vae-path "${VAE_PATH}" \
    --model DiT-L/2 \
    --resolutions 32,16,8 \
    --stage-lengths 300,300,400 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size "${TOTAL_SAMPLES}" \
    --vae-batch-size 8 \
    --rank 0 --world-size 1 --seed 0 --bf16 \
    > "${LOG_DIR}/rdd_unified_l_phased_dmd3_latest_3nfe.log" 2>&1 &
dmd_pid=$!

CUDA_VISIBLE_DEVICES=0 python -u 2d/sample_fid_unified.py \
    --config "${TWO_STAGE_CONFIG}" \
    --rank 0 --world-size 1 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size "${TOTAL_SAMPLES}" \
    --vae-batch-size 8 \
    --seed 0 \
    --vae-path "${VAE_PATH}" \
    --no-eval \
    > "${LOG_DIR}/rdd_unified_l_2stage_500500_step097169_ddpm250_cfg1.5.log" 2>&1 &
two_stage_pid=$!

failed=0
if ! wait "${dmd_pid}"; then
    echo "ERROR: DMD preview sampler failed" >&2
    failed=1
fi
if ! wait "${two_stage_pid}"; then
    echo "ERROR: two-stage preview sampler failed" >&2
    failed=1
fi

dmd_count="$(find "${DMD_OUTPUT}" -maxdepth 1 -type f -name '*.png' | wc -l)"
two_stage_output="${REPO_ROOT}/eval_results/previews/rdd_unified_l_2stage_500500_step097169_ddpm250_cfg1.5"
two_stage_count="$(find "${two_stage_output}" -maxdepth 1 -type f -name '*.png' | wc -l)"
echo "DMD preview images: ${dmd_count}/${TOTAL_SAMPLES} in ${DMD_OUTPUT}"
echo "Two-stage preview images: ${two_stage_count}/${TOTAL_SAMPLES} in ${two_stage_output}"

(( failed == 0 )) || exit 1
[[ "${dmd_count}" -eq "${TOTAL_SAMPLES}" ]]
[[ "${two_stage_count}" -eq "${TOTAL_SAMPLES}" ]]
