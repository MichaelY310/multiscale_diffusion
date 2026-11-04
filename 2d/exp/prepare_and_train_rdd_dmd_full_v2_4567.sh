#!/usr/bin/env bash
# Build validated deterministic pairs, enforce a visual/statistical gate, then train.

set -Eeuo pipefail
set -o pipefail

REPO_ROOT="${REPO_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PAIR_DIR="${PAIR_DIR:-${REPO_ROOT}/data/rdd_dmd_pairs/unified_l_300300400_cfg1.5_marginalddim250_v2}"
QUALITY_DIR="${QUALITY_DIR:-${REPO_ROOT}/eval_results/diagnostics/rdd_dmd_pair_quality_v2}"
VAE_PATH="${VAE_PATH:-${REPO_ROOT}/checkpoints/pretrained/stabilityai/sd-vae-ft-ema}"

cd "${REPO_ROOT}"
if [[ ! -s "${PAIR_DIR}/metadata.json" ]]; then
    OUTPUT_DIR="${PAIR_DIR}" \
        bash 2d/exp/build_rdd_dmd_regression_pairs_4567.sh
fi

if [[ -f "${REPO_ROOT}/.venv-rdd-eval/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-rdd-eval/bin/activate"
else
    echo "ERROR: missing ${REPO_ROOT}/.venv-rdd-eval" >&2
    exit 1
fi

CUDA_VISIBLE_DEVICES=4 python -u 2d/validate_rdd_dmd_regression_pairs.py \
    --pair-dir "${PAIR_DIR}" \
    --vae-path "${VAE_PATH}" \
    --output-dir "${QUALITY_DIR}" \
    --max-images 32 \
    --decode-batch 4

REGRESSION_DATA="${PAIR_DIR}" VAE_PATH="${VAE_PATH}" \
    bash 2d/exp/train_rdd_unified_l_phased_dmd3_dmdpaper_4567.sh
