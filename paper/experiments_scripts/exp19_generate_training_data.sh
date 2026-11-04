#!/bin/bash
# =============================================================================
# Experiment 19: Training Data Generation (80K Videos from Wan 2.1-14B)
# Purpose: Generate synthetic training data for RDD-Wan and PyramidalWan
# Hardware: 2 machines × 8×A100 each (16 GPUs total)
# Estimated time: ~7 days
# =============================================================================

set -e

WORK_DIR="/home/tiger/multiscale_diffusion"
VIDEO_DIR="${WORK_DIR}/rdd_video"
OUTPUT_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/training_data_80k"
PROMPTS_FILE="${VIDEO_DIR}/data/prompts_80k.txt"

# Teacher model
TEACHER_MODEL="Wan-AI/Wan2.1-T2V-14B"
NUM_VIDEOS=80000
VIDEOS_PER_GPU=5000  # 80000 / 16 GPUs
CFG_SCALE=5.0
NUM_STEPS=50
RESOLUTION="480p"
NUM_FRAMES=81

# Determine machine and GPU allocation
MACHINE_ID=${MACHINE_ID:-0}  # Set to 0 or 1
GPUS_PER_MACHINE=8
START_IDX=$((MACHINE_ID * GPUS_PER_MACHINE * VIDEOS_PER_GPU))

mkdir -p ${OUTPUT_DIR}/raw_videos
mkdir -p ${OUTPUT_DIR}/latents
mkdir -p ${OUTPUT_DIR}/metadata

echo "=== Experiment 19: Training Data Generation ==="
echo "Machine: ${MACHINE_ID}, Start index: ${START_IDX}"
echo "Teacher: ${TEACHER_MODEL}"
echo "Target: ${NUM_VIDEOS} videos total"

# --- Step 1: Generate prompts if not exist ---
if [ ! -f "${PROMPTS_FILE}" ]; then
    echo "[$(date)] Generating prompt list..."
    python ${VIDEO_DIR}/scripts/prepare_prompts.py \
        --source sharegpt4video \
        --num-prompts ${NUM_VIDEOS} \
        --output ${PROMPTS_FILE} \
        --diverse-categories
fi

# --- Step 2: Generate videos with Ulysses Sequence Parallel ---
# Each GPU generates its assigned portion
echo "[$(date)] Starting video generation with Ulysses SP..."

# Using 4-GPU sequence parallel (SP=4) means 2 parallel generation jobs per machine
# Each job uses 4 GPUs via SP, generating VIDEOS_PER_GPU*4 videos
NUM_SP_GROUPS=2  # 8 GPUs / 4 SP degree = 2 parallel groups

for GROUP_ID in $(seq 0 $((NUM_SP_GROUPS - 1))); do
    GROUP_START=$((START_IDX + GROUP_ID * VIDEOS_PER_GPU * 4))
    GROUP_END=$((GROUP_START + VIDEOS_PER_GPU * 4))
    GPU_OFFSET=$((GROUP_ID * 4))
    
    echo "[$(date)] Launching SP group ${GROUP_ID} (videos ${GROUP_START}-${GROUP_END})..."
    
    CUDA_VISIBLE_DEVICES="${GPU_OFFSET},$((GPU_OFFSET+1)),$((GPU_OFFSET+2)),$((GPU_OFFSET+3))" \
    torchrun --nproc_per_node=4 \
        ${VIDEO_DIR}/generate_training_data.py \
        --model-path ${TEACHER_MODEL} \
        --prompts-file ${PROMPTS_FILE} \
        --start-idx ${GROUP_START} \
        --end-idx ${GROUP_END} \
        --cfg-scale ${CFG_SCALE} \
        --num-steps ${NUM_STEPS} \
        --resolution ${RESOLUTION} \
        --num-frames ${NUM_FRAMES} \
        --output-dir ${OUTPUT_DIR}/raw_videos \
        --save-latents ${OUTPUT_DIR}/latents \
        --ulysses-sp-degree 4 \
        --save-metadata ${OUTPUT_DIR}/metadata \
        2>&1 | tee ${OUTPUT_DIR}/generation_machine${MACHINE_ID}_group${GROUP_ID}.log &
done

# Wait for all groups to finish
wait
echo "[$(date)] All generation jobs complete for machine ${MACHINE_ID}"

# --- Step 3: Verify and filter ---
echo "[$(date)] Verifying generated videos..."
python ${VIDEO_DIR}/scripts/verify_training_data.py \
    --data-dir ${OUTPUT_DIR} \
    --expected-count $((VIDEOS_PER_GPU * GPUS_PER_MACHINE)) \
    --check-corrupt \
    --min-motion-score 0.1 \
    --output-stats ${OUTPUT_DIR}/stats_machine${MACHINE_ID}.json

echo "=== Experiment 19 (Machine ${MACHINE_ID}) Complete ==="
echo "Generated videos in: ${OUTPUT_DIR}/raw_videos"
echo "Pre-encoded latents in: ${OUTPUT_DIR}/latents"
