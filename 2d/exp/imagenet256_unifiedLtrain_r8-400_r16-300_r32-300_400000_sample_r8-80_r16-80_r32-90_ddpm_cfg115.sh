#!/bin/bash

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
CONFIG="${SCRIPT_DIR}/imagenet256_unifiedL300300400_400000_908080step_ddpm_cfg115.jsonl"

echo "======================================"
echo " UnifiedL 300-300-400 Sampling (400K)"
echo " Config: ${CONFIG}"
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

echo "All sampling done!"
