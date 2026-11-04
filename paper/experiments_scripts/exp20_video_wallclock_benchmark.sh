#!/bin/bash
# =============================================================================
# Experiment 20: Wall-Clock Latency Benchmark (Video Inference)
# Purpose: Measure actual inference time for all methods
# Hardware: Single A100 80GB
# Estimated time: ~4 hours
# =============================================================================

set -e

WORK_DIR="/home/tiger/multiscale_diffusion"
VIDEO_DIR="${WORK_DIR}/rdd_video"
OUTPUT_DIR="${WORK_DIR}/logs/video_experiments/exp20_wallclock"
REMOTE_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video_checkpoints"

RDD_WAN_CKPT="${REMOTE_CKPT}/rdd_wan_5k.pt"
PYRAMIDAL_WAN_CKPT="${REMOTE_CKPT}/pyramidal_wan_5k.pt"
WAN_VANILLA_CKPT="Wan-AI/Wan2.1-T2V-1.3B"

NUM_WARMUP=3
NUM_MEASURE=10
PROMPT="A cat walking gracefully across a sunlit garden"

mkdir -p ${OUTPUT_DIR}

echo "=== Experiment 20: Wall-Clock Latency Benchmark ==="
echo "Warmup runs: ${NUM_WARMUP}, Measured runs: ${NUM_MEASURE}"

# --- Benchmark function ---
benchmark_method() {
    local METHOD=$1
    local CKPT=$2
    local SOLVER=$3
    local NFE=$4
    local FRAMES=$5
    local TAG="${METHOD}_${SOLVER}_nfe${NFE}_f${FRAMES}"
    
    echo "[$(date)] Benchmarking: ${TAG}"
    
    python ${VIDEO_DIR}/benchmark_inference.py \
        --model-path ${CKPT} \
        --method ${METHOD} \
        --solver ${SOLVER} \
        --num-steps ${NFE} \
        --cfg-scale 5.0 \
        --resolution 480p \
        --num-frames ${FRAMES} \
        --prompt "${PROMPT}" \
        --num-warmup ${NUM_WARMUP} \
        --num-measure ${NUM_MEASURE} \
        --output ${OUTPUT_DIR}/${TAG}.json \
        --report-per-step \
        --report-memory \
        2>&1 | tee ${OUTPUT_DIR}/${TAG}.log
}

# --- 81 frames benchmarks ---
echo "--- 81 frames ---"
for NFE in 20 30 50; do
    benchmark_method "wan_vanilla" ${WAN_VANILLA_CKPT} "euler" ${NFE} 81
done

for NFE in 20 30 50; do
    benchmark_method "pyramidal_wan" ${PYRAMIDAL_WAN_CKPT} "euler" ${NFE} 81
done

for NFE in 20 30 50; do
    benchmark_method "rdd_wan" ${RDD_WAN_CKPT} "euler" ${NFE} 81
done

for NFE in 20 30 50; do
    benchmark_method "rdd_wan" ${RDD_WAN_CKPT} "dpm_solver_2" ${NFE} 81
done

# --- Scaling analysis: different frame counts ---
echo "--- Frame scaling analysis ---"
for FRAMES in 33 81 161; do
    benchmark_method "wan_vanilla" ${WAN_VANILLA_CKPT} "euler" 50 ${FRAMES}
    benchmark_method "rdd_wan" ${RDD_WAN_CKPT} "dpm_solver_2" 30 ${FRAMES}
done

# --- Aggregate results ---
python -c "
import json, glob, os
import numpy as np

results = []
for f in sorted(glob.glob('${OUTPUT_DIR}/*.json')):
    with open(f) as fp:
        data = json.load(fp)
    results.append(data)

print('\\n' + '='*80)
print('WALL-CLOCK BENCHMARK RESULTS')
print('='*80)
print(f'{\"Method\":<35} {\"NFE\":<5} {\"Frames\":<7} {\"Latency(s)\":<12} {\"Speedup\":<10} {\"Memory(GB)\":<10}')
print('-'*80)

# Find vanilla baseline for speedup calculation
vanilla_50_81 = next((r for r in results if 'wan_vanilla' in r.get('tag','') and r.get('nfe')==50 and r.get('frames')==81), None)
baseline_latency = vanilla_50_81['mean_latency'] if vanilla_50_81 else 1.0

for r in results:
    tag = r.get('tag', os.path.basename(r.get('file','')))
    nfe = r.get('nfe', '?')
    frames = r.get('frames', '?')
    latency = r.get('mean_latency', 0)
    memory = r.get('peak_memory_gb', 0)
    speedup = baseline_latency / latency if latency > 0 else 0
    print(f'{tag:<35} {nfe:<5} {frames:<7} {latency:<12.2f} {speedup:<10.2f}x {memory:<10.1f}')

print('='*80)
"

echo "=== Experiment 20 Complete ==="
echo "Results in: ${OUTPUT_DIR}"
