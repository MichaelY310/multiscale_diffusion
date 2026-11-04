#!/bin/bash
# 等待 RDD-FM-Unified-XL-v2 的 0400000.pt 出现后自动上传到 HF
# 用法: 在 tmux 中运行 bash /home/tiger/multiscale_diffusion/wait_and_upload_fm_400k.sh

set +e

CKPT_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-FM-Unified-XL-v2/002-imagenet256-fm-unified/checkpoints"
TARGET="0400000.pt"
TARGET_PATH="${CKPT_DIR}/${TARGET}"

REPO_ID="yang335/search_agent_hl_to_i18n_transport"
HF_PATH="checkpoints/RDD-FM-Unified-XL-v2/checkpoints/${TARGET}"

# 网络配置
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

echo "============================================"
echo " Waiting for: ${TARGET_PATH}"
echo " Will upload to: ${REPO_ID} -> ${HF_PATH}"
echo "============================================"

# 轮询等待文件出现
while [ ! -f "$TARGET_PATH" ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for ${TARGET}..."
    sleep 60
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] File appeared! Waiting 120s for write to complete..."
sleep 120

# 确认文件大小稳定（不再增长）
SIZE1=$(stat -c%s "$TARGET_PATH" 2>/dev/null || echo 0)
sleep 30
SIZE2=$(stat -c%s "$TARGET_PATH" 2>/dev/null || echo 0)
while [ "$SIZE1" != "$SIZE2" ]; do
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] File still writing (${SIZE1} -> ${SIZE2}), waiting..."
    SIZE1=$SIZE2
    sleep 30
    SIZE2=$(stat -c%s "$TARGET_PATH" 2>/dev/null || echo 0)
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] File stable at ${SIZE2} bytes. Uploading..."

# 上传
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate

hf upload "$REPO_ID" "$TARGET_PATH" "$HF_PATH" --commit-message "Upload RDD-FM-Unified-XL-v2 ${TARGET}"

if [ $? -eq 0 ]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ Upload successful!"
else
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ Upload failed! Retrying in 60s..."
    sleep 60
    hf upload "$REPO_ID" "$TARGET_PATH" "$HF_PATH" --commit-message "Upload RDD-FM-Unified-XL-v2 ${TARGET}"
fi

echo "Done."
