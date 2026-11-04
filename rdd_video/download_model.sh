#!/bin/bash
# ============================================================================
# Download Wan 2.1 T2V 1.3B model
# ============================================================================
#
# Prerequisites:
#   pip install huggingface_hub
#
# Usage:
#   bash download_model.sh [MODEL_DIR]
#
# Default download location:
#   /home/tiger/multiscale_diffusion/pretrained_models/Wan2.1-T2V-1.3B
#
# ============================================================================

set -e

MODEL_DIR="${1:-/home/tiger/multiscale_diffusion/pretrained_models/Wan2.1-T2V-1.3B}"
mkdir -p "$MODEL_DIR"

echo "============================================"
echo " Wan 2.1 T2V 1.3B Model Downloader"
echo "============================================"
echo ""
echo "Target directory: $MODEL_DIR"
echo ""

# ============================================================================
# Method 1: HuggingFace Hub (recommended)
# ============================================================================
echo "Downloading Wan 2.1 T2V 1.3B from HuggingFace..."
echo ""

huggingface-cli download Wan-AI/Wan2.1-T2V-1.3B \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False

echo ""
echo "Download complete: $MODEL_DIR"
echo ""
echo "Contents:"
ls -lh "$MODEL_DIR"

echo ""
echo "============================================"
echo " Download finished successfully!"
echo "============================================"
echo ""
echo "You can now use this model with:"
echo "  python generate_training_data.py --pretrained-dir $MODEL_DIR ..."
echo ""

# ============================================================================
# ALTERNATIVE METHODS (uncomment if HuggingFace is not accessible)
# ============================================================================

# --- Method 2: Git LFS clone from HuggingFace ---
# Requires: git lfs install
#
# git lfs install
# git clone https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B "$MODEL_DIR"
#

# --- Method 3: ModelScope (for users in China) ---
# Requires: pip install modelscope
#
# python -c "
# from modelscope.hub.snapshot_download import snapshot_download
# snapshot_download(
#     'Wan-AI/Wan2.1-T2V-1.3B',
#     cache_dir='$MODEL_DIR',
# )
# "
#
# Or using modelscope CLI:
# modelscope download --model Wan-AI/Wan2.1-T2V-1.3B --local_dir "$MODEL_DIR"
#

# --- Method 4: Manual download ---
# If automated tools fail, manually download from:
#   https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/tree/main
#
# Required files:
#   - config.json
#   - diffusion_pytorch_model*.safetensors (diffusion model weights)
#   - vae/ (VAE model directory)
#   - text_encoder/ (T5 text encoder)
#   - tokenizer/ (T5 tokenizer)
#
# Place all files in: $MODEL_DIR
#

# --- Method 5: wget/aria2c direct links ---
# For faster multi-connection download:
#
# BASE_URL="https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B/resolve/main"
# aria2c -x 16 -s 16 -d "$MODEL_DIR" "$BASE_URL/config.json"
# aria2c -x 16 -s 16 -d "$MODEL_DIR" "$BASE_URL/diffusion_pytorch_model.safetensors"
# # ... repeat for all files
#
