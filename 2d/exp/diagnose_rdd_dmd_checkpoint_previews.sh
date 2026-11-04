#!/usr/bin/env bash

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
GPU="${GPU:-0}"
TOTAL_SAMPLES="${TOTAL_SAMPLES:-8}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"
DMD_CKPT_DIR="${REPO_ROOT}/results_rdd_dmd3/unified_l_300300400_400k_phased_dmd3_dmd_imagenet_gb256/checkpoints"
TEACHER_CKPT="${REPO_ROOT}/checkpoints/RDD-Unified-L/0400000.pt"
OUTPUT_ROOT="${REPO_ROOT}/eval_results/diagnostics/rdd_dmd_checkpoint_previews"
LOG_ROOT="${OUTPUT_ROOT}/logs"

cd "${REPO_ROOT}"
source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

run_preview() {
    local checkpoint="$1"
    local weight_source="$2"
    local tag="$3"
    local output_dir="${OUTPUT_ROOT}/${tag}"
    mkdir -p "${output_dir}"
    CUDA_VISIBLE_DEVICES="${GPU}" python -u 2d/sample_rdd_phased_dmd3.py \
        --checkpoint "${checkpoint}" \
        --weight-source "${weight_source}" \
        --output-dir "${output_dir}" \
        --vae-path "${VAE_PATH}" \
        --model DiT-L/2 \
        --resolutions 32,16,8 \
        --stage-lengths 300,300,400 \
        --total-samples "${TOTAL_SAMPLES}" \
        --batch-size "${TOTAL_SAMPLES}" \
        --vae-batch-size 8 \
        --rank 0 --world-size 1 --seed 0 --bf16 \
        > "${LOG_ROOT}/${tag}.log" 2>&1
    local count
    count="$(find "${output_dir}" -maxdepth 1 -type f -name '*.png' | wc -l)"
    [[ "${count}" -eq "${TOTAL_SAMPLES}" ]] || {
        echo "ERROR: ${tag} generated ${count}/${TOTAL_SAMPLES}" >&2
        return 1
    }
    echo "${tag}: ${count}/${TOTAL_SAMPLES}"
}

# Same seeds and labels for every row.  The teacher row is the exact
# initialization of both the DMD student and fake-score model.
run_preview "${TEACHER_CKPT}" ema teacher_init_400k
run_preview "${DMD_CKPT_DIR}/0005000.pt" ema dmd_step005000_ema
run_preview "${DMD_CKPT_DIR}/0050000.pt" ema dmd_step050000_ema
run_preview "${DMD_CKPT_DIR}/latest.pt" ema dmd_latest_ema
run_preview "${DMD_CKPT_DIR}/latest.pt" student dmd_latest_raw_student

echo "All DMD checkpoint previews complete: ${OUTPUT_ROOT}"
