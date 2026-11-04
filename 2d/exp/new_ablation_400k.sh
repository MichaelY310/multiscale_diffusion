#!/bin/bash
# New ablation experiments for paper (400K checkpoints):
#   1. DPM-Solver-2 vs DDPM at matched NFE (Pareto curve) — key RDD advantage
#   2. Stage time-axis split sensitivity (300-300-400 vs 250-250-500 vs 200-300-500 vs 400-300-300)
#   3. CFG scale sweep (1.5, 2.0, 2.5, 4.0)
# Total: 16 experiments

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
CONFIG="${SCRIPT_DIR}/new_ablation_400k.jsonl"

echo "======================================"
echo " New Ablation Experiments (400K step)"
echo " Config: ${CONFIG}"
echo " Experiments: 16 total"
echo "   - DPM-Solver-2 NFE curve (uni+ms): 6 configs"
echo "   - DDPM low-step baseline:           3 configs"
echo "   - DPM-Solver-2 full steps:          1 config"
echo "   - Stage split sensitivity:          3 configs"
echo "   - CFG scale sweep:                  3 configs"
echo " GPUs: ${NUM_GPUS}"
echo "======================================"

for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u "${SAMPLE_SCRIPT}" \
        --config "${CONFIG}" \
        --rank $RANK --world-size $NUM_GPUS \
        --total-samples 50000 --batch-size 256 \
        --seed 0 &
done
wait

echo "All new ablation experiments done!"
