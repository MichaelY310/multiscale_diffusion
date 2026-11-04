#!/bin/bash
# 计算 exp04 DiT convergence 各 checkpoint 的 FID
# 包含 exp04_convergence 下新跑的 + 之前已有采样的 400K/500K

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

# 格式: "目录全路径|输出名"
ALL_JOBS=(
    "${EVAL_BASE}/dit_100K|dit_100K"
    "${EVAL_BASE}/dit_200K|dit_200K"
    "${EVAL_BASE}/dit_300K|dit_300K"
    "${EVAL_BASE}/dit_50K|dit_50K"
    "${EVAL_BASE}/dit_150K|dit_150K"
    "${EVAL_BASE}/dit_250K|dit_250K"
    "${EVAL_BASE}/dit_350K|dit_350K"
    "${EVAL_BASE}/dit_450K|dit_450K"
    "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/eval_mse/eval_dit400k_250step_ddpm_nocfg_mse|dit_400K"
    "/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp_samples_original/dit500k_250step_ddpm_cfg1.0|dit_500K"
)

echo "==========================================================="
echo " Computing FID for ${#ALL_JOBS[@]} DiT experiments"
echo "==========================================================="

for ENTRY in "${ALL_JOBS[@]}"; do
    FULL_PATH="${ENTRY%%|*}"
    TAG="${ENTRY##*|}"
    LOCAL_NPZ="/tmp/${TAG}_samples.npz"
    LOG_FILE="${LOG_DIR}/${TAG}.txt"

    echo ""
    echo "-----------------------------------------------------------"
    echo "  Processing: ${TAG}"
    echo "  Source: ${FULL_PATH}"
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
for ENTRY in "${ALL_JOBS[@]}"; do
    TAG="${ENTRY##*|}"
    LOG_FILE="${LOG_DIR}/${TAG}.txt"
    if [ -f "$LOG_FILE" ]; then
        FID=$(grep "^FID:" "$LOG_FILE" 2>/dev/null | head -1)
        echo "  ${TAG}: ${FID:-NOT COMPUTED}"
    else
        echo "  ${TAG}: NOT COMPUTED"
    fi
done
