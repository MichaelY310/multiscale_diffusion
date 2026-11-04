#!/bin/bash
# =============================================================================
# Experiment 18: Video NFE Pareto Curve
# Purpose: Full NFE sweep to generate Pareto frontier plots
# Hardware: 8×A100
# Estimated time: ~216 GPU-hours (36 VBench evaluations)
# =============================================================================

set -e

WORK_DIR="/home/tiger/multiscale_diffusion"
VIDEO_DIR="${WORK_DIR}/rdd_video"
OUTPUT_DIR="${WORK_DIR}/logs/video_experiments/exp18_nfe_pareto"
REMOTE_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/video_checkpoints"

RDD_WAN_CKPT="${REMOTE_CKPT}/rdd_wan_5k.pt"
PYRAMIDAL_WAN_CKPT="${REMOTE_CKPT}/pyramidal_wan_5k.pt"
WAN_VANILLA_CKPT="Wan-AI/Wan2.1-T2V-1.3B"

NFE_VALUES=(10 15 20 25 30 40 50 75 100)

mkdir -p ${OUTPUT_DIR}

echo "=== Experiment 18: NFE Pareto Curve ==="
echo "NFE sweep: ${NFE_VALUES[*]}"

run_single_eval() {
    local METHOD=$1
    local CKPT=$2
    local SOLVER=$3
    local NFE=$4
    local TAG="${METHOD}_${SOLVER}_nfe${NFE}"
    local OUT="${OUTPUT_DIR}/${TAG}"
    
    if [ -f "${OUT}/vbench_results/results.json" ]; then
        echo "[SKIP] ${TAG} already evaluated"
        return
    fi
    
    mkdir -p ${OUT}
    echo "[$(date)] ${TAG} starting..."
    
    torchrun --nproc_per_node=8 \
        ${VIDEO_DIR}/sample_video_rdd.py \
        --model-path ${CKPT} \
        --method ${METHOD} \
        --solver ${SOLVER} \
        --num-steps ${NFE} \
        --cfg-scale 5.0 \
        --resolution 480p \
        --num-frames 81 \
        --prompts-source vbench_standard \
        --output-dir ${OUT}/videos \
        2>&1 | tee ${OUT}/generation.log
    
    python -m vbench.evaluate \
        --videos-dir ${OUT}/videos \
        --output-dir ${OUT}/vbench_results \
        --full-suite \
        2>&1 | tee ${OUT}/vbench_eval.log
}

# Wan vanilla (Euler) - full sweep
for NFE in "${NFE_VALUES[@]}"; do
    run_single_eval "wan_vanilla" ${WAN_VANILLA_CKPT} "euler" ${NFE}
done

# PyramidalWan (Euler only)
for NFE in "${NFE_VALUES[@]}"; do
    run_single_eval "pyramidal_wan" ${PYRAMIDAL_WAN_CKPT} "euler" ${NFE}
done

# RDD-Wan (Euler)
for NFE in "${NFE_VALUES[@]}"; do
    run_single_eval "rdd_wan" ${RDD_WAN_CKPT} "euler" ${NFE}
done

# RDD-Wan (DPM-Solver-2) - the key advantage
for NFE in "${NFE_VALUES[@]}"; do
    run_single_eval "rdd_wan" ${RDD_WAN_CKPT} "dpm_solver_2" ${NFE}
done

# --- Plot Pareto curve ---
python -c "
import json, glob, os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

methods = {
    'wan_vanilla_euler': {'label': 'Wan 2.1 (Euler)', 'color': 'gray', 'marker': 'o'},
    'pyramidal_wan_euler': {'label': 'PyramidalWan (Euler)', 'color': 'blue', 'marker': 's'},
    'rdd_wan_euler': {'label': 'RDD-Wan (Euler)', 'color': 'orange', 'marker': '^'},
    'rdd_wan_dpm_solver_2': {'label': 'RDD-Wan (DPM-2)', 'color': 'red', 'marker': 'D'},
}

fig, ax = plt.subplots(1, 1, figsize=(8, 5))

for method_prefix, style in methods.items():
    nfes, scores = [], []
    for d in sorted(glob.glob(f'${OUTPUT_DIR}/{method_prefix}_nfe*/vbench_results/results.json')):
        nfe = int(os.path.basename(os.path.dirname(os.path.dirname(d))).split('nfe')[1])
        with open(d) as f:
            data = json.load(f)
        nfes.append(nfe)
        scores.append(data.get('total_score', 0))
    
    if nfes:
        order = np.argsort(nfes)
        ax.plot(np.array(nfes)[order], np.array(scores)[order], 
                label=style['label'], color=style['color'], marker=style['marker'], linewidth=2)

ax.set_xlabel('NFE (Number of Function Evaluations)', fontsize=12)
ax.set_ylabel('VBench Total Score', fontsize=12)
ax.set_title('NFE vs Quality Pareto Frontier (Video Generation)', fontsize=13)
ax.legend(fontsize=10)
ax.set_xscale('log')
ax.grid(True, alpha=0.3)
ax.axvspan(15, 30, alpha=0.1, color='green', label='Practical zone')

plt.tight_layout()
plt.savefig('${OUTPUT_DIR}/pareto_curve.pdf', dpi=150, bbox_inches='tight')
plt.savefig('${OUTPUT_DIR}/pareto_curve.png', dpi=150, bbox_inches='tight')
print('Pareto curve saved to ${OUTPUT_DIR}/pareto_curve.pdf')
"

echo "=== Experiment 18 Complete ==="
