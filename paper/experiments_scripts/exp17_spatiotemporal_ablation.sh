#!/bin/bash
# =============================================================================
# Experiment 17: Spatiotemporal Drift Ablation
# Purpose: Disentangle spatial vs temporal vs joint drift contributions
# Hardware: 8×A100
# Estimated time: ~80 hours (4 training runs × 5K steps + VBench evals)
# =============================================================================

set -e

WORK_DIR="/home/tiger/multiscale_diffusion"
VIDEO_DIR="${WORK_DIR}/rdd_video"
OUTPUT_DIR="${WORK_DIR}/logs/video_experiments/exp17_ablation"
REMOTE_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video_checkpoints"
DATA_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/training_data_80k"

WAN_PRETRAINED="Wan-AI/Wan2.1-T2V-1.3B"
TRAIN_STEPS=5000
LR=1e-5
BATCH_SIZE=1  # per GPU, effective = 8

mkdir -p ${OUTPUT_DIR}

echo "=== Experiment 17: Spatiotemporal Drift Ablation ==="

# --- Variant 1: Spatial-only drift ---
echo "[$(date)] Training spatial-only variant..."
torchrun --nproc_per_node=8 \
    ${VIDEO_DIR}/train_rdd_wan.py \
    --pretrained ${WAN_PRETRAINED} \
    --data-dir ${DATA_DIR} \
    --output-dir ${REMOTE_CKPT}/ablation_spatial_only \
    --drift-type spatial_only \
    --spatial-factors "4,2,1" \
    --temporal-factors "1,1,1" \
    --stage-boundaries "0.06,0.41" \
    --num-steps ${TRAIN_STEPS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    2>&1 | tee ${OUTPUT_DIR}/train_spatial_only.log

# --- Variant 2: Temporal-only drift ---
echo "[$(date)] Training temporal-only variant..."
torchrun --nproc_per_node=8 \
    ${VIDEO_DIR}/train_rdd_wan.py \
    --pretrained ${WAN_PRETRAINED} \
    --data-dir ${DATA_DIR} \
    --output-dir ${REMOTE_CKPT}/ablation_temporal_only \
    --drift-type temporal_only \
    --spatial-factors "1,1,1" \
    --temporal-factors "4,2,1" \
    --stage-boundaries "0.06,0.41" \
    --num-steps ${TRAIN_STEPS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    2>&1 | tee ${OUTPUT_DIR}/train_temporal_only.log

# --- Variant 3: Joint drift (default RDD-Wan) ---
echo "[$(date)] Training joint drift variant..."
torchrun --nproc_per_node=8 \
    ${VIDEO_DIR}/train_rdd_wan.py \
    --pretrained ${WAN_PRETRAINED} \
    --data-dir ${DATA_DIR} \
    --output-dir ${REMOTE_CKPT}/ablation_joint \
    --drift-type joint \
    --spatial-factors "4,2,1" \
    --temporal-factors "4,2,1" \
    --stage-boundaries "0.06,0.41" \
    --num-steps ${TRAIN_STEPS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    2>&1 | tee ${OUTPUT_DIR}/train_joint.log

# --- Evaluate all variants on VBench ---
for VARIANT in spatial_only temporal_only joint; do
    echo "[$(date)] Evaluating ${VARIANT}..."
    CKPT="${REMOTE_CKPT}/ablation_${VARIANT}/checkpoint_final.pt"
    
    torchrun --nproc_per_node=8 \
        ${VIDEO_DIR}/sample_video_rdd.py \
        --model-path ${CKPT} \
        --method rdd_wan \
        --solver dpm_solver_2 \
        --num-steps 30 \
        --cfg-scale 5.0 \
        --resolution 480p \
        --num-frames 81 \
        --prompts-source vbench_standard \
        --output-dir ${OUTPUT_DIR}/${VARIANT}/videos \
        2>&1 | tee ${OUTPUT_DIR}/${VARIANT}/generation.log
    
    python -m vbench.evaluate \
        --videos-dir ${OUTPUT_DIR}/${VARIANT}/videos \
        --output-dir ${OUTPUT_DIR}/${VARIANT}/vbench_results \
        --full-suite \
        2>&1 | tee ${OUTPUT_DIR}/${VARIANT}/vbench_eval.log
done

echo "=== Experiment 17 Complete ==="
echo "Results in: ${OUTPUT_DIR}"
