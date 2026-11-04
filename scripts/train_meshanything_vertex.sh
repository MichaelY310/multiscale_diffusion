#!/bin/bash
# ==============================================================================
# RDD-DiT-3D Vertex 训练完整 Pipeline (v5: token 过滤 + attention mask 修复)
# ==============================================================================
#
# 优化点:
#   - 模型: DiT3D-RDD-B (768/12/12, ~110M params) 适配 4216 样本
#   - max-tokens 27476: 过滤掉 P99 以上极端样本 (~42个), 避免超长序列拖慢训练
#   - dataset: 全向量化 compute_stage_data (无 Python for 循环)
#   - model: attention mask 恢复 (正确处理 padding), memory-efficient attention
#   - gradient-checkpointing + bf16 AMP
#   - batch_size=2, grad_accum=4 (等效 BS=64)
#   - LR: 2e-4 + warmup 500
#
# ==============================================================================

set -e

# ── 环境设置 ──
export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT

cd /home/tiger/multiscale_diffusion

# ── 路径配置 ──
DATA_ROOT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/MeshAnythingV2"
ST_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/set_transformer/checkpoints_emd_ae_concat_3d/step_8400000.pt"
OUTPUT_ROOT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/RDD-DiT-3D"
PREPROC_DIR="${OUTPUT_ROOT}/preprocessed_R256_K10"
TRAIN_DIR="${OUTPUT_ROOT}/training_vertex_R256_v5"

# GPU 数量 (自动检测)
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
if [ "$NUM_GPUS" -lt 1 ]; then
    NUM_GPUS=1
fi
echo "检测到 ${NUM_GPUS} 张 GPU"

# ==============================================================================
# Step 1: 预处理 (跳过如果已存在)
# ==============================================================================
echo "================================================================"
echo "Step 1: 预处理 MeshAnythingV2 → 稀疏 embedding"
echo "================================================================"

if [ ! -d "$PREPROC_DIR" ] || [ $(ls "$PREPROC_DIR"/*.pt 2>/dev/null | wc -l) -lt 100 ]; then
    python RDD_DiT_3D/preprocess_mesh_to_sparse.py \
        --input "${DATA_ROOT}/test_merged.npz" \
        --ckpt "$ST_CKPT" \
        --output-dir "$PREPROC_DIR" \
        --resolution 256 \
        --k-max 10 \
        --filter-by-kmax \
        --batch-size 512 \
        --device cuda:0
else
    echo "预处理数据已存在 ($PREPROC_DIR)，跳过。"
    echo "  文件数: $(ls $PREPROC_DIR/*.pt | wc -l)"
fi

# ==============================================================================
# Step 2: 训练 RDD-DiT-3D
# ==============================================================================
echo ""
echo "================================================================"
echo "Step 2: 训练 RDD-DiT-3D (unified mode, 6 stages)"
echo "================================================================"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=29500 \
    RDD_DiT_3D/train.py \
    --data-dir "$PREPROC_DIR" \
    --results-dir "$TRAIN_DIR" \
    --model DiT3D-RDD-B \
    --in-channels 256 \
    --num-classes 1 \
    --class-dropout-prob 0.0 \
    --base-size 8 \
    --stage-resolutions 8,16,32,64,128,256 \
    --steps-per-stage 100,150,150,200,200,200 \
    --diffusion-steps 1000 \
    --schedule squaredcos_cap_v2 \
    --mode unified \
    --epochs 5000 \
    --batch-size 2 \
    --lr 2e-4 \
    --wd 0.01 \
    --warmup-steps 500 \
    --min-lr 1e-5 \
    --total-steps 20000 \
    --grad-accum-steps 4 \
    --gradient-checkpointing \
    --amp \
    --ema-decay 0.9999 \
    --log-every 50 \
    --ckpt-every 2000 \
    --max-tokens 27476

echo ""
echo "================================================================"
echo "训练完成!"
echo "================================================================"
