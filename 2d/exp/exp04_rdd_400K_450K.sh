#!/bin/bash
# 补跑 rdd_400K 和 rdd_450K 的采样 + FID 计算

set +e
set -o pipefail

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/2d

export HF_HUB_OFFLINE=1
NUM_GPUS=8

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
EVAL_BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp04_convergence"
CKPT_BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints"

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

run_rdd() {
    local STEP=$1
    local TAG=$2
    local CKPT="${CKPT_BASE}/${STEP}.pt"
    local OUTDIR="${EVAL_BASE}/${TAG}"
    local LOG_FILE="${LOG_DIR}/${TAG}.txt"

    echo ""
    echo "============================================================"
    echo " ${TAG}: sampling + FID"
    echo " Checkpoint: ${CKPT}"
    echo " Output: ${OUTDIR}"
    echo "============================================================"

    # 采样
    local EXISTING=$(find "$OUTDIR" -name "*.png" 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge 50000 ]; then
        echo "  Already has ${EXISTING} images, skip sampling."
    else
        mkdir -p "$OUTDIR"
        # 写临时 jsonl
        local TMP_CFG="/tmp/rdd_${TAG}.jsonl"
        echo "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-XL/2\",\"ckpt\":\"${CKPT}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.0,\"upsample_mode\":\"fresh\",\"output_dir\":\"${OUTDIR}\"}" > "$TMP_CFG"

        for RANK in $(seq 0 $((NUM_GPUS-1))); do
            CUDA_VISIBLE_DEVICES=$RANK python -u "${SAMPLE_SCRIPT}" \
                --config "${TMP_CFG}" \
                --rank $RANK --world-size $NUM_GPUS \
                --total-samples 50000 --batch-size 256 \
                --seed 0 &
        done
        wait
        echo "  Sampling done."
    fi

    # FID 计算
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

run_rdd "0400000" "rdd_400K"
run_rdd "0450000" "rdd_450K"

echo ""
echo "All done!"
