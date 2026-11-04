#!/bin/bash
# 补完 dit_150K 和 dit_350K 的采样 + FID 计算

set +e
set -o pipefail

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/DiT

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

EVAL_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp04_convergence
CKPT_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/DiT-XL-2
TOTAL=50000
BS=256
WS=8

REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
PNG_TO_NPZ="/home/tiger/multiscale_diffusion/2d/png_to_npz.py"
LOG_DIR="/home/tiger/multiscale_diffusion/2d/exp/eval_results_exp04"
mkdir -p "$LOG_DIR"

metrics_complete() {
    local file="$1" metric
    [ -f "$file" ] || return 1
    for metric in "Inception Score" "FID" "sFID" "Precision" "Recall"; do
        grep -q "^${metric}:" "$file" || return 1
    done
}

run_dit() {
    local CKPT=$1
    local TAG=$2

    local OUTDIR="${EVAL_BASE}/${TAG}"

    local EXISTING=$(find "$OUTDIR" -name "*.png" 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge "$TOTAL" ]; then
        echo "=== SKIP ${TAG}: already has ${EXISTING} images ==="
    else
        echo ""
        echo "============================================================"
        echo " Resuming: ${TAG}, DDPM 250step, CFG=1.0"
        echo " Checkpoint: ${CKPT}"
        echo " Output: ${OUTDIR} (existing: ${EXISTING})"
        echo "============================================================"

        mkdir -p "$OUTDIR"

        for RANK in $(seq 0 $((WS-1))); do
            CUDA_VISIBLE_DEVICES=$RANK python -u sample_fid.py \
                --ckpt $CKPT --model DiT-XL/2 --image-size 256 --num-classes 1000 --vae ema \
                --cfg-scale 1.0 --num-sampling-steps 250 --sampling-method ddpm \
                --total-samples $TOTAL --batch-size $BS --seed 0 \
                --rank $RANK --world-size $WS \
                --output-dir $OUTDIR &
        done
        wait
        echo "=== Sampling done: ${TAG} ==="
    fi

    # FID 计算
    local LOG_FILE="${LOG_DIR}/${TAG}.txt"
    if metrics_complete "$LOG_FILE"; then
        echo "  All five metrics already computed:"
        grep -E '^(Inception Score|FID|sFID|Precision|Recall):' "$LOG_FILE"
        return
    fi

    local FILE_COUNT=$(find "$OUTDIR" -maxdepth 1 -name "*.png" -type f | wc -l)
    if [ "$FILE_COUNT" -lt 49900 ]; then
        echo "  Not enough files ($FILE_COUNT), skipping FID"
        return
    fi

    local LOCAL_NPZ="/tmp/${TAG}_samples.npz"
    echo "  [1/2] Packing PNG -> NPZ..."
    python "$PNG_TO_NPZ" "$OUTDIR" --num-samples 50000 --output "$LOCAL_NPZ"
    if [ $? -ne 0 ]; then
        echo "  NPZ packing failed"
        rm -f "$LOCAL_NPZ"
        return
    fi

    echo "  [2/2] Computing FID..."
    python -u "$EVALUATOR" "$REF_NPZ" "$LOCAL_NPZ" 2>&1 | tee "$LOG_FILE"
    echo "  Result:"
    grep "^FID\|^sFID\|^Precision\|^Recall" "$LOG_FILE"
    rm -f "$LOCAL_NPZ"
}

run_dit "${CKPT_BASE}/004-DiT-XL-2/checkpoints/0150000.pt" "dit_150K"
run_dit "${CKPT_BASE}/006-DiT-XL-2/checkpoints/0350000.pt" "dit_350K"

echo ""
echo "All done!"
