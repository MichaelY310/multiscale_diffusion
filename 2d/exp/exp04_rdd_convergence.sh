#!/bin/bash
# exp04 机器A: RDD convergence curve (cfg=1.0, ddpm 80+80+90, noise_filling)
# 8 configs: 100K, 200K, 300K (优先), 50K, 150K, 250K, 350K, 500K (补充)

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/2d

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
CONFIG="${SCRIPT_DIR}/exp04_rdd_convergence.jsonl"

echo "======================================"
echo " exp04: RDD Convergence Curve"
echo " Config: ${CONFIG}"
echo " Experiments: 8 (100K,200K,300K first, then 50K,150K,250K,350K,500K)"
echo " GPUs: ${NUM_GPUS}, batch_size: 256"
echo "======================================"

for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u "${SAMPLE_SCRIPT}" \
        --config "${CONFIG}" \
        --rank $RANK --world-size $NUM_GPUS \
        --total-samples 50000 --batch-size 256 \
        --seed 0 &
done
wait

echo "exp04 RDD convergence done!"
