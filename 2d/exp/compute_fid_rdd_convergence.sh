#!/bin/bash
# 计算 exp04 RDD convergence 各 checkpoint 的 FID

set +e
set -o pipefail

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate

REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
PNG_TO_NPZ="/home/tiger/multiscale_diffusion/2d/png_to_npz.py"
EVAL_BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp04_convergence"
LOG_DIR="/home/tiger/multiscale_diffusion/2d/exp/eval_results_exp04"

mkdir -p "$LOG_DIR"

metrics_complete() {
    local file="$1" metric
    [ -f "$file" ] || return 1
    for metric in "Inception Score" "FID" "sFID" "Precision" "Recall"; do
        grep -q "^${metric}:" "$file" || return 1
    done
}

DIRS=(
    "rdd_100K"
    "rdd_200K"
    "rdd_300K"
    "rdd_50K"
    "rdd_150K"
    "rdd_250K"
    "rdd_350K"
    "rdd_500K"
)

echo "==========================================================="
echo " Computing FID for ${#DIRS[@]} RDD checkpoints"
echo "==========================================================="

for DIR_NAME in "${DIRS[@]}"; do
    FULL_PATH="${EVAL_BASE}/${DIR_NAME}"
    LOCAL_NPZ="/tmp/${DIR_NAME}_samples.npz"
    LOG_FILE="${LOG_DIR}/${DIR_NAME}.txt"

    echo ""
    echo "-----------------------------------------------------------"
    echo "  Processing: ${DIR_NAME}"
    echo "-----------------------------------------------------------"

    # 跳过已评估
    if metrics_complete "$LOG_FILE"; then
        echo "  Already evaluated, skipping. Result:"
        grep "^FID\|^sFID\|^IS\|^Precision\|^Recall" "$LOG_FILE"
        continue
    fi

    # 检查目录
    if [ ! -d "$FULL_PATH" ]; then
        echo "  Directory not found: $FULL_PATH"
        continue
    fi

    # 检查文件数量
    FILE_COUNT=$(find "$FULL_PATH" -maxdepth 1 -name "*.png" -type f | wc -l)
    echo "  Found $FILE_COUNT PNG files"
    if [ "$FILE_COUNT" -lt 49900 ]; then
        echo "  Not enough files (need >=49900), skipping"
        continue
    fi

    # Step 1: 打包 NPZ
    if [ ! -f "$LOCAL_NPZ" ]; then
        echo "  [1/2] Packing PNG -> NPZ..."
        python "$PNG_TO_NPZ" "$FULL_PATH" --num-samples 50000 --output "$LOCAL_NPZ"
        if [ $? -ne 0 ]; then
            echo "  NPZ packing failed"
            rm -f "$LOCAL_NPZ"
            continue
        fi
    else
        echo "  [1/2] NPZ already exists at $LOCAL_NPZ"
    fi

    # Step 2: 计算 FID
    echo "  [2/2] Computing FID..."
    python -u "$EVALUATOR" "$REF_NPZ" "$LOCAL_NPZ" 2>&1 | tee "$LOG_FILE"
    if [ $? -ne 0 ]; then
        echo "  Evaluation failed"
        continue
    fi

    echo "  Done! Result:"
    grep "^FID\|^sFID\|^IS\|^Precision\|^Recall" "$LOG_FILE"

    # 清理 /tmp 节省空间
    rm -f "$LOCAL_NPZ"
done

echo ""
echo "==========================================================="
echo " All done! Results in: $LOG_DIR"
echo "==========================================================="
echo ""
echo "Summary:"
for DIR_NAME in "${DIRS[@]}"; do
    LOG_FILE="${LOG_DIR}/${DIR_NAME}.txt"
    if [ -f "$LOG_FILE" ]; then
        FID=$(grep "^FID:" "$LOG_FILE" 2>/dev/null | head -1)
        echo "  ${DIR_NAME}: ${FID:-NOT COMPUTED}"
    fi
done
