#!/bin/bash
# =============================================================================
# Experiment 16: RDD-Wan vs PyramidalWan VBench Full Evaluation
# Purpose: Compare RDD-Wan vs PyramidalWan on VBench (946 prompts)
# Hardware: 8×A100
# Estimated time: ~48 hours total (6h per VBench eval × 8 configs)
# =============================================================================

set -e

# --- Configuration ---
WORK_DIR="/home/tiger/multiscale_diffusion"
VIDEO_DIR="${WORK_DIR}/rdd_video"
OUTPUT_DIR="${WORK_DIR}/logs/video_experiments/exp16_vbench"
REMOTE_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video_checkpoints"

# Model checkpoints (after exp19 training)
RDD_WAN_CKPT="${REMOTE_CKPT}/rdd_wan_5k.pt"
PYRAMIDAL_WAN_CKPT="${REMOTE_CKPT}/pyramidal_wan_5k.pt"
WAN_VANILLA_CKPT="Wan-AI/Wan2.1-T2V-1.3B"

# VBench config
VBENCH_PROMPTS="vbench_standard"  # 946 prompts
RESOLUTION="480p"  # 848x480
NUM_FRAMES=81
CFG_SCALE=5.0

mkdir -p ${OUTPUT_DIR}

echo "=== Experiment 16: VBench Full Evaluation ==="
echo "Start time: $(date)"

# --- Helper function ---
run_vbench_eval() {
    local METHOD=$1
    local CKPT=$2
    local SOLVER=$3
    local NFE=$4
    local OUT_SUBDIR="${OUTPUT_DIR}/${METHOD}_${SOLVER}_nfe${NFE}"
    
    mkdir -p ${OUT_SUBDIR}
    
    echo "[$(date)] Running: ${METHOD} | Solver: ${SOLVER} | NFE: ${NFE}"
    
    # Step 1: Generate videos
    torchrun --nproc_per_node=8 \
        ${VIDEO_DIR}/sample_video_rdd.py \
        --model-path ${CKPT} \
        --method ${METHOD} \
        --solver ${SOLVER} \
        --num-steps ${NFE} \
        --cfg-scale ${CFG_SCALE} \
        --resolution ${RESOLUTION} \
        --num-frames ${NUM_FRAMES} \
        --prompts-source ${VBENCH_PROMPTS} \
        --output-dir ${OUT_SUBDIR}/videos \
        2>&1 | tee ${OUT_SUBDIR}/generation.log
    
    # Step 2: Run VBench evaluation
    python -m vbench.evaluate \
        --videos-dir ${OUT_SUBDIR}/videos \
        --output-dir ${OUT_SUBDIR}/vbench_results \
        --full-suite \
        2>&1 | tee ${OUT_SUBDIR}/vbench_eval.log
    
    echo "[$(date)] Done: ${METHOD}_${SOLVER}_nfe${NFE}"
}

# --- Run all configurations ---
# Wan 2.1 vanilla (baseline)
for NFE in 20 30 50; do
    run_vbench_eval "wan_vanilla" ${WAN_VANILLA_CKPT} "euler" ${NFE}
done

# PyramidalWan (Euler only - DPM incompatible)
for NFE in 20 30 50; do
    run_vbench_eval "pyramidal_wan" ${PYRAMIDAL_WAN_CKPT} "euler" ${NFE}
done

# RDD-Wan (Euler)
for NFE in 20 30 50; do
    run_vbench_eval "rdd_wan" ${RDD_WAN_CKPT} "euler" ${NFE}
done

# RDD-Wan (DPM-Solver-2) - THE KEY COMPARISON
for NFE in 20 30 50; do
    run_vbench_eval "rdd_wan" ${RDD_WAN_CKPT} "dpm_solver_2" ${NFE}
done

echo "=== Experiment 16 Complete ==="
echo "End time: $(date)"
echo "Results in: ${OUTPUT_DIR}"

# --- Aggregate results ---
python -c "
import json, glob, os
results = {}
for d in sorted(glob.glob('${OUTPUT_DIR}/*/vbench_results/results.json')):
    name = os.path.basename(os.path.dirname(os.path.dirname(d)))
    with open(d) as f:
        data = json.load(f)
    results[name] = {
        'quality_score': data.get('quality_score', 'N/A'),
        'semantic_score': data.get('semantic_score', 'N/A'),
        'total_score': data.get('total_score', 'N/A')
    }
print('\\n=== SUMMARY ===')
for k, v in results.items():
    print(f'{k}: Total={v[\"total_score\"]:.4f}, Quality={v[\"quality_score\"]:.4f}, Semantic={v[\"semantic_score\"]:.4f}')
"
