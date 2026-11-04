#!/bin/bash
# RDD-Multistage-100-300-600 400K: cfg=1.0 + cfg=1.5, ddpm 80+80+90, noise_filling
# 采样 + FID 计算

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/2d

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
CONFIG="${SCRIPT_DIR}/ms100300600_400k_noisefilling.jsonl"

REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
PNG_TO_NPZ="/home/tiger/multiscale_diffusion/2d/png_to_npz.py"
RESULT_DIR="/home/tiger/multiscale_diffusion/2d/eval_results_unified"
mkdir -p "$RESULT_DIR"

echo "======================================"
echo " RDD-Multistage-100-300-600 400K"
echo " cfg=1.0 + cfg=1.5, ddpm 80+80+90, noise_filling"
echo " Config: ${CONFIG}"
echo "======================================"

# 采样
for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u "${SAMPLE_SCRIPT}" \
        --config "${CONFIG}" \
        --rank $RANK --world-size $NUM_GPUS \
        --total-samples 50000 --batch-size 256 \
        --seed 0 &
done
wait

echo "=== Sampling done! Computing FID... ==="

# FID 计算
EVAL_BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/eval_mse"

compute_fid() {
    local TAG=$1
    local OUTDIR="${EVAL_BASE}/${TAG}"
    local LOG_FILE="${RESULT_DIR}/${TAG}.txt"

    echo ""
    echo "-----------------------------------------------------------"
    echo "  Computing FID: ${TAG}"
    echo "-----------------------------------------------------------"

    local FILE_COUNT=$(find "$OUTDIR" -maxdepth 1 -name "*.png" -type f | wc -l)
    if [ "$FILE_COUNT" -lt 49900 ]; then
        echo "  Not enough files ($FILE_COUNT), skipping"
        return
    fi

    local LOCAL_NPZ="/tmp/${TAG}_samples.npz"
    echo "  [1/2] Packing PNG -> NPZ..."
    python "$PNG_TO_NPZ" "$OUTDIR" --num-samples 50000 --output "$LOCAL_NPZ"

    echo "  [2/2] Computing FID..."
    python -u "$EVALUATOR" "$REF_NPZ" "$LOCAL_NPZ" 2>&1 | tee "$LOG_FILE"
    echo "  Done! Result:"
    grep "^FID\|^sFID\|^Precision\|^Recall" "$LOG_FILE"
    rm -f "$LOCAL_NPZ"
}

compute_fid "ms100300600_400k_808090_ddpm_cfg1_noisefilling"
compute_fid "ms100300600_400k_808090_ddpm_cfg15_noisefilling"

echo ""
echo "==========================================================="
echo " All done!"
echo "==========================================================="
