#!/bin/bash
# exp11_extended_training.sh
# ==========================
# 继续训练 RDD Unified XL 到 750K 步（从 400K checkpoint resume）。
# 每 50K 步做一次评估（在 450K, 500K, 550K, 600K, 650K, 700K, 750K 处）。
#
# 模型: DiT-XL/2 (Unified), batch 256, 8×A100
# Stage lengths: 300,300,400 (default XL config)
# 评估: 50K 采样, cfg=1.5, DPM-2
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp11_extended_training.sh

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
TRAIN_SCRIPT="/home/tiger/multiscale_diffusion/2d/train_RDD_imagenet256_unified.py"
SAMPLE_SCRIPT="/home/tiger/multiscale_diffusion/2d/sample_fid_unified.py"
DATA_PATH="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified"
RESUME_CKPT="${RESULTS_DIR}/checkpoints/0400000.pt"

export HF_HOME="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface"

# ─────────────────────────────────────────────────────────────────────────────
# Training arguments
# ─────────────────────────────────────────────────────────────────────────────
MODEL="DiT-XL/2"
RESOLUTIONS="32,16,8"
STAGE_LENGTHS="300,300,400"
BATCH_SIZE=256
DIFFUSION_STEPS=1000
IMAGE_SIZE=256
DATA_SOURCE="imagenet"
CKPT_EVERY=5000
LOG_EVERY=100
NUM_WORKERS=4

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Resume training from 400K to 750K
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Resuming training from 400K to 750K steps"
echo "  Resume checkpoint: ${RESUME_CKPT}"
echo "============================================================"

torchrun --nproc_per_node=8 ${TRAIN_SCRIPT} \
    --model ${MODEL} \
    --resolutions ${RESOLUTIONS} \
    --stage-lengths ${STAGE_LENGTHS} \
    --results-dir ${RESULTS_DIR} \
    --data-path ${DATA_PATH} \
    --data-source ${DATA_SOURCE} \
    --image-size ${IMAGE_SIZE} \
    --diffusion-steps ${DIFFUSION_STEPS} \
    --global-batch-size ${BATCH_SIZE} \
    --ckpt-every ${CKPT_EVERY} \
    --log-every ${LOG_EVERY} \
    --num-workers ${NUM_WORKERS} \
    --resume ${RESUME_CKPT}

echo "Training to 750K complete."

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate at every 50K checkpoint (450K, 500K, ..., 750K)
# ─────────────────────────────────────────────────────────────────────────────
echo "============================================================"
echo "Running evaluations at 50K intervals (450K-750K)"
echo "============================================================"

EVAL_STEPS=(450000 500000 550000 600000 650000 700000 750000)

for STEP in "${EVAL_STEPS[@]}"; do
    STEP_STR=$(printf "%07d" ${STEP})
    CKPT_PATH="${RESULTS_DIR}/checkpoints/${STEP_STR}.pt"

    if [ ! -f "${CKPT_PATH}" ]; then
        echo "WARNING: Checkpoint not found: ${CKPT_PATH}, skipping..."
        continue
    fi

    echo "------------------------------------------------------------"
    echo "Evaluating step ${STEP} (${CKPT_PATH})"
    echo "------------------------------------------------------------"

    SAMPLE_CONFIG="/tmp/exp11_eval_${STEP_STR}.jsonl"
    cat > ${SAMPLE_CONFIG} << EOF
{"name":"RDD-XL-extended-${STEP_STR}","model_type":"unified","model":"DiT-XL/2","ckpt":"${CKPT_PATH}","resolutions":"32,16,8","stage_lengths":"300,300,400","cfg_scale":1.5,"sampler":"dpm2","diffusion_steps":1000}
EOF

    python -u ${SAMPLE_SCRIPT} \
        --config ${SAMPLE_CONFIG} \
        --rank 0 --world-size 8 \
        --total-samples 50000 --batch-size 100 \
        --seed 0

    echo "Evaluation at step ${STEP} complete."
done

echo "============================================================"
echo "All extended training evaluations done."
echo "============================================================"
