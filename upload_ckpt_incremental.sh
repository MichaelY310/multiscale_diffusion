#!/bin/bash
# 修正上传：上传到 HF 上已有的正确路径
# HF 结构:
#   checkpoints/DiT-XL-2/0500000.pt (扁平)
#   checkpoints/RDD-FM-Unified-XL-v2/checkpoints/0350000.pt
#   checkpoints/RDD-Multistage-100-300-600/checkpoints_*/0350000.pt
#   checkpoints/RDD-Unified-L/0500000.pt (扁平)

set -e

# 网络配置
export http_proxy=http://sys-proxy-rd-relay.byted.org:8118
export https_proxy=http://sys-proxy-rd-relay.byted.org:8118
export HTTP_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export HTTPS_PROXY=http://sys-proxy-rd-relay.byted.org:8118
export no_proxy=localhost,127.0.0.1
export HF_HUB_DISABLE_XET=1

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate

REPO="yang335/search_agent_hl_to_i18n_transport"
BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original"
STAGING="/tmp/hf_ckpt_upload_$$"
rm -rf "$STAGING"
mkdir -p "$STAGING"

# --- 1. DiT-XL-2 500K (扁平: checkpoints/DiT-XL-2/0500000.pt) ---
SRC="${BASE}/DiT-XL-2/007-DiT-XL-2/checkpoints/0500000.pt"
DST="${STAGING}/checkpoints/DiT-XL-2/0500000.pt"
mkdir -p "$(dirname "$DST")"
ln -sf "$SRC" "$DST"

# --- 2. RDD-FM-Unified-XL-v2: 350K + 370K ---
#    HF 路径: checkpoints/RDD-FM-Unified-XL-v2/checkpoints/0350000.pt
for STEP in 0350000 0370000; do
    SRC="${BASE}/RDD-FM-Unified-XL-v2/002-imagenet256-fm-unified/checkpoints/${STEP}.pt"
    DST="${STAGING}/checkpoints/RDD-FM-Unified-XL-v2/checkpoints/${STEP}.pt"
    mkdir -p "$(dirname "$DST")"
    ln -sf "$SRC" "$DST"
done

# --- 3. RDD-Multistage-100-300-600: 350K, 400K (3 resolutions) ---
#    HF 路径: checkpoints/RDD-Multistage-100-300-600/checkpoints_8x8/0350000.pt
for RES in checkpoints_8x8 checkpoints_16x16 checkpoints_32x32; do
    for STEP in 0350000 0400000; do
        SRC="${BASE}/RDD-Multistage-100-300-600/003-imagenet256-multiscale/${RES}/${STEP}.pt"
        DST="${STAGING}/checkpoints/RDD-Multistage-100-300-600/${RES}/${STEP}.pt"
        mkdir -p "$(dirname "$DST")"
        ln -sf "$SRC" "$DST"
    done
done

# --- 4. RDD-Unified-L: 500K + 535K (扁平: checkpoints/RDD-Unified-L/0500000.pt) ---
for STEP in 0500000 0535000; do
    SRC="${BASE}/RDD-Unified-L-300-300-400/imagenet256-unified/checkpoints/${STEP}.pt"
    DST="${STAGING}/checkpoints/RDD-Unified-L/${STEP}.pt"
    mkdir -p "$(dirname "$DST")"
    ln -sf "$SRC" "$DST"
done

echo "=== Staging complete. Files to upload: ==="
find "$STAGING" -type l | sort
echo ""
echo "Total: $(find "$STAGING" -type l | wc -l) files"
echo ""

echo "Uploading to ${REPO}..."
hf upload "$REPO" "$STAGING" . --commit-message "Fix paths: DiT-500K, FM-XL-v2 350K/370K, MS-100-300-600 350K/400K, Uni-L 500K/535K"

echo "Done! Cleaning up staging..."
rm -rf "$STAGING"
echo "Upload complete."
