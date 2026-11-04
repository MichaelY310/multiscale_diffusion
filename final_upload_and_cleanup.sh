#!/bin/bash
# 最终清理脚本：等待所有任务完成 → 上传 → 清理 → 杀 worker
# 在 tmux 中运行: bash /home/tiger/multiscale_diffusion/final_upload_and_cleanup.sh

set +e

# ═══════════════════════════════════════════════════════════════
# 网络配置
# ═══════════════════════════════════════════════════════════════
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate

REPO_ID="yang335/search_agent_hl_to_i18n_transport"

# 结果文件路径
DIT_250K_RESULT="/home/tiger/multiscale_diffusion/2d/exp/eval_results_exp04/dit_250K.txt"
DIT_450K_RESULT="/home/tiger/multiscale_diffusion/2d/exp/eval_results_exp04/dit_450K.txt"

# ckpt 路径
CKPT_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Unified-XL-v2/002-imagenet256-fm-unified/checkpoints"
CKPT_TARGET="${CKPT_DIR}/0400000.pt"

echo "============================================"
echo " Final Upload & Cleanup Script"
echo " Started: $(date)"
echo "============================================"

# ═══════════════════════════════════════════════════════════════
# TASK 1: 等待 dit_250K 和 dit_450K 的 FID 结果
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[TASK 1] Waiting for dit_250K and dit_450K FID results..."

while true; do
    DONE=true
    if [ ! -f "$DIT_250K_RESULT" ] || ! grep -q "^FID:" "$DIT_250K_RESULT" 2>/dev/null; then
        DONE=false
        echo "  [$(date '+%H:%M:%S')] dit_250K.txt not ready yet"
    fi
    if [ ! -f "$DIT_450K_RESULT" ] || ! grep -q "^FID:" "$DIT_450K_RESULT" 2>/dev/null; then
        DONE=false
        echo "  [$(date '+%H:%M:%S')] dit_450K.txt not ready yet"
    fi
    if [ "$DONE" = true ]; then
        echo "  [$(date '+%H:%M:%S')] Both FID results ready!"
        grep "^FID" "$DIT_250K_RESULT"
        grep "^FID" "$DIT_450K_RESULT"
        break
    fi
    sleep 60
done

# ═══════════════════════════════════════════════════════════════
# TASK 2: 等待 0400000.pt checkpoint
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[TASK 2] Waiting for ${CKPT_TARGET}..."

while [ ! -f "$CKPT_TARGET" ]; do
    echo "  [$(date '+%H:%M:%S')] Waiting for 0400000.pt..."
    sleep 60
done

echo "  [$(date '+%H:%M:%S')] File appeared! Waiting 120s for write to complete..."
sleep 120

# 确认文件大小稳定
SIZE1=$(stat -c%s "$CKPT_TARGET" 2>/dev/null || echo 0)
sleep 30
SIZE2=$(stat -c%s "$CKPT_TARGET" 2>/dev/null || echo 0)
while [ "$SIZE1" != "$SIZE2" ]; do
    echo "  [$(date '+%H:%M:%S')] File still writing (${SIZE1} -> ${SIZE2}), waiting..."
    SIZE1=$SIZE2
    sleep 30
    SIZE2=$(stat -c%s "$CKPT_TARGET" 2>/dev/null || echo 0)
done
echo "  [$(date '+%H:%M:%S')] Checkpoint stable at ${SIZE2} bytes."

# ═══════════════════════════════════════════════════════════════
# TASK 3: 上传所有日志 + FID 结果
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[TASK 3] Uploading all logs..."
bash /home/tiger/multiscale_diffusion/upload_logs_only.sh
echo "  Logs upload done."

# ═══════════════════════════════════════════════════════════════
# TASK 4: 上传 0400000.pt checkpoint
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[TASK 4] Uploading 0400000.pt..."

HF_CKPT_PATH="checkpoints/RDD-FM-Unified-XL-v2/checkpoints/0400000.pt"
hf upload "$REPO_ID" "$CKPT_TARGET" "$HF_CKPT_PATH" --commit-message "Upload RDD-FM-Unified-XL-v2 0400000.pt"

if [ $? -eq 0 ]; then
    echo "  [$(date '+%H:%M:%S')] ✓ Checkpoint upload successful!"
else
    echo "  [$(date '+%H:%M:%S')] ✗ Upload failed, retrying..."
    sleep 60
    hf upload "$REPO_ID" "$CKPT_TARGET" "$HF_CKPT_PATH" --commit-message "Upload RDD-FM-Unified-XL-v2 0400000.pt"
fi

# ═══════════════════════════════════════════════════════════════
# TASK 5: 清理远端数据
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[TASK 5] Cleaning up remote data..."
rm -rf /mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD
echo "  Remote data deleted."

# ═══════════════════════════════════════════════════════════════
# TASK 6: 清理本地 + 杀 worker
# ═══════════════════════════════════════════════════════════════
echo ""
echo "[TASK 6] Cleaning local and killing workers..."
rm -rf /home/tiger/*
mlx worker kill 4125605 4125606 4125607 4125620

echo ""
echo "============================================"
echo " ALL DONE. $(date)"
echo "============================================"
