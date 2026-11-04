#!/bin/bash
# ============================================================================
# RDD-Wan 完整数据准备脚本
# Step 1: 下载 Wan 2.1-T2V-1.3B 模型
# Step 2: 安装 Wan 依赖
# Step 3: 批量生成 80K 合成训练视频（8卡并行）
# ============================================================================

set -e

# ─── 路径配置 ──────────────────────────────────────────────────────────────
BASE_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video"
MODEL_DIR="${BASE_DIR}/Wan2.1-T2V-1.3B"
DATA_DIR="${BASE_DIR}/synthetic_data"
WAN_REPO_DIR="/home/tiger/multiscale_diffusion/Wan2.1"
SCRIPT_DIR="/home/tiger/multiscale_diffusion/rdd_video"

export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

mkdir -p "$BASE_DIR"
mkdir -p "$DATA_DIR"

echo "============================================"
echo "  RDD-Wan Data Preparation Pipeline"
echo "============================================"
echo "Model dir:  $MODEL_DIR"
echo "Data dir:   $DATA_DIR"
echo "Wan repo:   $WAN_REPO_DIR"
echo ""

# ============================================================================
# Step 1: 下载 Wan 2.1-T2V-1.3B 模型
# ============================================================================
echo "============================================"
echo " Step 1: Downloading Wan 2.1 T2V 1.3B"
echo "============================================"

if [ -f "${MODEL_DIR}/config.json" ]; then
    echo "Model already exists at ${MODEL_DIR}, skipping download."
else
    pip install -U huggingface_hub 2>/dev/null

    python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    'Wan-AI/Wan2.1-T2V-1.3B',
    local_dir='${MODEL_DIR}',
    local_dir_use_symlinks=False,
)
print('Download complete!')
"

    echo "Model downloaded to: $MODEL_DIR"
fi

echo ""

# ============================================================================
# Step 2: 安装 Wan 代码库
# ============================================================================
echo "============================================"
echo " Step 2: Installing Wan"
echo "============================================"

if python -c "import wan" 2>/dev/null; then
    echo "Wan already installed, skipping."
else
    if [ ! -d "$WAN_REPO_DIR" ]; then
        echo "Cloning Wan2.1 repository..."
        cd /home/tiger/multiscale_diffusion
        git clone https://github.com/Wan-Video/Wan2.1.git
    fi

    echo "Installing Wan..."
    cd "$WAN_REPO_DIR"
    pip install -e . 2>&1 | tail -5
    echo "Wan installed."
fi

echo ""

# ============================================================================
# Step 3: Download prompts (Open-Sora Plan 350K subset)
# ============================================================================
echo "============================================"
echo " Step 3: Preparing prompts"
echo "============================================"

PROMPTS_FILE="${BASE_DIR}/prompts.txt"

if [ -f "$PROMPTS_FILE" ] && [ $(wc -l < "$PROMPTS_FILE") -gt 100 ]; then
    echo "Prompts file already exists ($(wc -l < $PROMPTS_FILE) lines), skipping."
else
    echo "Generating diverse prompts from built-in set (will cycle to target count)..."
    echo "Using built-in 50 prompts, cycled to 2000 for initial experiment."
    # 使用脚本内置的 prompt（先用 2000 条做快速验证，之后可换 350K）
    touch "$PROMPTS_FILE"
fi

echo ""

# ============================================================================
# Step 4: 生成合成视频（8 GPU 并行）
# ============================================================================
echo "============================================"
echo " Step 4: Generating synthetic training data"
echo "============================================"
echo ""
echo "Ready to generate. Run the following commands on 8 GPUs:"
echo ""
echo "  # Option A: 2000 samples for quick validation (default prompts)"
echo "  # Estimated time: ~5 hours on 8xH100"
echo ""

NUM_GPUS=8
NUM_SAMPLES=2000

for i in $(seq 0 $((NUM_GPUS - 1))); do
    echo "  CUDA_VISIBLE_DEVICES=$i python ${SCRIPT_DIR}/generate_training_data.py \\"
    echo "      --pretrained-dir ${MODEL_DIR} \\"
    echo "      --output-dir ${DATA_DIR} \\"
    echo "      --num-samples ${NUM_SAMPLES} \\"
    echo "      --size 832x448 \\"
    echo "      --frame-num 81 \\"
    echo "      --num-gpus ${NUM_GPUS} --gpu-rank $i \\"
    echo "      --device 0 &"
    echo ""
done

echo "  wait"
echo ""
echo "  # Option B: Full 80K (PyramidalWan scale)"
echo "  # Change --num-samples 80000"
echo ""

echo "============================================"
echo " Setup complete!"
echo "============================================"
