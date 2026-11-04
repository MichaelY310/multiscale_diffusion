#!/usr/bin/env bash
# Build the 25K fixed noise/teacher-output pairs used by original DMD ImageNet.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
TEACHER_CHECKPOINT="${TEACHER_CHECKPOINT:-${REPO_ROOT}/checkpoints/RDD-Unified-L/0400000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/data/rdd_dmd_pairs/unified_l_300300400_cfg1.5_marginalddim250_v2}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/logs/build_rdd_dmd_pairs_unified_l_300300400_cfg1.5_marginalddim250_v2.log}"
MASTER_PORT="${MASTER_PORT:-29651}"

cd "${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/.venv-rdd-eval/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
elif [[ -f "${REPO_ROOT}/2d/2d/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/2d/2d/bin/activate"
else
    echo "ERROR: RDD Python environment not found" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" "$(dirname "${LOG_FILE}")"
echo "Building original-DMD fixed pairs: 25,000, CFG=1.5, validated RDD marginal-DDIM250"
echo "Noise stored explicitly: 8x8 initial + 16x16/32x32 boundary noise"
echo "Output: ${OUTPUT_DIR}"

CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun \
    --standalone --master_port "${MASTER_PORT}" --nproc_per_node 4 \
    2d/build_rdd_dmd_regression_pairs.py \
    --teacher-checkpoint "${TEACHER_CHECKPOINT}" \
    --output-dir "${OUTPUT_DIR}" \
    --model DiT-L/2 \
    --resolutions 32,16,8 \
    --stage-lengths 300,300,400 \
    --sampling-steps 80,80,90 \
    --teacher-cfg 1.5 \
    --total-pairs 25000 \
    --batch-per-gpu 16 \
    --seed 0 2>&1 | tee -a "${LOG_FILE}"
