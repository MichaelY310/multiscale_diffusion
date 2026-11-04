#!/bin/bash
# exp10_stage_boundary_train.sh
# ==============================
# 训练2个额外的 stage boundary 配置，用于消融实验：
#   配置A: [200, 300, 500] (T_1=500, T_2=800)
#   配置B: [400, 300, 300] (T_1=300, T_2=600)
#
# 模型: DiT-XL/2 (Unified), batch 256, 8×A100
# 训练: 400K 步
# 评估: 50K 采样, cfg=1.5, DPM-2
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp10_stage_boundary_train.sh

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="/home/tiger/multiscale_diffusion/2d/train_RDD_imagenet256_unified.py"
SAMPLE_SCRIPT="/home/tiger/multiscale_diffusion/2d/sample_fid_unified.py"
DATA_PATH="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"
RESULTS_BASE="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original"

export HF_HOME="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface"

# ─────────────────────────────────────────────────────────────────────────────
# Common training arguments
# ─────────────────────────────────────────────────────────────────────────────
MODEL="DiT-XL/2"
RESOLUTIONS="32,16,8"
BATCH_SIZE=256
DIFFUSION_STEPS=1000
IMAGE_SIZE=256
DATA_SOURCE="imagenet"
CKPT_EVERY=5000
LOG_EVERY=100
NUM_WORKERS=4

# ─────────────────────────────────────────────────────────────────────────────
# Configuration A: stage-lengths 200,300,500
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[Config A] Training with stage-lengths 200,300,500"
echo "============================================================"

STAGE_A="200,300,500"
RESULTS_DIR_A="${RESULTS_BASE}/RDD-Unified-XL-200-300-500/imagenet256-unified"

torchrun --nproc_per_node=8 ${TRAIN_SCRIPT} \
    --model ${MODEL} \
    --resolutions ${RESOLUTIONS} \
    --stage-lengths ${STAGE_A} \
    --results-dir ${RESULTS_DIR_A} \
    --data-path ${DATA_PATH} \
    --data-source ${DATA_SOURCE} \
    --image-size ${IMAGE_SIZE} \
    --diffusion-steps ${DIFFUSION_STEPS} \
    --global-batch-size ${BATCH_SIZE} \
    --ckpt-every ${CKPT_EVERY} \
    --log-every ${LOG_EVERY} \
    --num-workers ${NUM_WORKERS}

echo "[Config A] Training complete. Starting evaluation..."

# Create JSONL config for sampling
CKPT_A="${RESULTS_DIR_A}/checkpoints/0400000.pt"
SAMPLE_CONFIG_A="/tmp/exp10_config_a.jsonl"
cat > ${SAMPLE_CONFIG_A} << EOF
{"name":"RDD-XL-200-300-500","model_type":"unified","model":"DiT-XL/2","ckpt":"${CKPT_A}","resolutions":"32,16,8","stage_lengths":"200,300,500","cfg_scale":1.5,"sampler":"dpm2","diffusion_steps":1000}
EOF

python -u ${SAMPLE_SCRIPT} \
    --config ${SAMPLE_CONFIG_A} \
    --rank 0 --world-size 8 \
    --total-samples 50000 --batch-size 100 \
    --seed 0

echo "[Config A] Evaluation complete."

# ─────────────────────────────────────────────────────────────────────────────
# Configuration B: stage-lengths 400,300,300
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "[Config B] Training with stage-lengths 400,300,300"
echo "============================================================"

STAGE_B="400,300,300"
RESULTS_DIR_B="${RESULTS_BASE}/RDD-Unified-XL-400-300-300/imagenet256-unified"

torchrun --nproc_per_node=8 ${TRAIN_SCRIPT} \
    --model ${MODEL} \
    --resolutions ${RESOLUTIONS} \
    --stage-lengths ${STAGE_B} \
    --results-dir ${RESULTS_DIR_B} \
    --data-path ${DATA_PATH} \
    --data-source ${DATA_SOURCE} \
    --image-size ${IMAGE_SIZE} \
    --diffusion-steps ${DIFFUSION_STEPS} \
    --global-batch-size ${BATCH_SIZE} \
    --ckpt-every ${CKPT_EVERY} \
    --log-every ${LOG_EVERY} \
    --num-workers ${NUM_WORKERS}

echo "[Config B] Training complete. Starting evaluation..."

# Create JSONL config for sampling
CKPT_B="${RESULTS_DIR_B}/checkpoints/0400000.pt"
SAMPLE_CONFIG_B="/tmp/exp10_config_b.jsonl"
cat > ${SAMPLE_CONFIG_B} << EOF
{"name":"RDD-XL-400-300-300","model_type":"unified","model":"DiT-XL/2","ckpt":"${CKPT_B}","resolutions":"32,16,8","stage_lengths":"400,300,300","cfg_scale":1.5,"sampler":"dpm2","diffusion_steps":1000}
EOF

python -u ${SAMPLE_SCRIPT} \
    --config ${SAMPLE_CONFIG_B} \
    --rank 0 --world-size 8 \
    --total-samples 50000 --batch-size 100 \
    --seed 0

echo "[Config B] Evaluation complete."
echo "============================================================"
echo "All stage boundary experiments done."
echo "============================================================"
