#!/usr/bin/env bash
# Generate a small, deterministic preview from the latest exact RDD-DMD model.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-4}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-16}"
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_rdd_dmd_full_v2_gb256/checkpoints/latest.pt}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
RUN_TAG="${RUN_TAG:-$(TZ=America/Los_Angeles date +%Y%m%d_la%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/eval_results/diagnostics/rdd_dmd_latest_preview/${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/sample_rdd_dmd_latest_preview_${RUN_TAG}.log}"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"
[[ -s "${CHECKPOINT}" ]] || { echo "ERROR: checkpoint not found: ${CHECKPOINT}" >&2; exit 1; }

echo "checkpoint=${CHECKPOINT}" | tee "${LOG_FILE}"
echo "output=${OUTPUT_DIR}; GPU=${GPU}; samples=${TOTAL_SAMPLES}" | tee -a "${LOG_FILE}"
CUDA_VISIBLE_DEVICES="${GPU}" python -u 2d/sample_rdd_phased_dmd3.py \
    --checkpoint "${CHECKPOINT}" \
    --weight-source ema \
    --output-dir "${OUTPUT_DIR}" \
    --vae-path "${VAE_PATH}" \
    --model DiT-L/2 \
    --resolutions 32,16,8 \
    --stage-lengths 300,300,400 \
    --total-samples "${TOTAL_SAMPLES}" \
    --batch-size 8 \
    --vae-batch-size 8 \
    --rank 0 --world-size 1 --seed 0 --bf16 \
    2>&1 | tee -a "${LOG_FILE}"

count="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.png' | wc -l)"
[[ "${count}" -eq "${TOTAL_SAMPLES}" ]] || {
    echo "ERROR: generated ${count}/${TOTAL_SAMPLES} PNG files" | tee -a "${LOG_FILE}" >&2
    exit 1
}
echo "COMPLETE: ${count} PNG files in ${OUTPUT_DIR}" | tee -a "${LOG_FILE}"
