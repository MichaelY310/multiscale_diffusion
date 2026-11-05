#!/bin/bash
# Multistage 300-300-400, 380K step, 90-80-80 DDPM, cfg=1.0 & 1.5
# 8-GPU parallel

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
CONFIG="${SCRIPT_DIR}/imagenet256_multistage300300400_400000_908080step_ddpm_cfg115.jsonl"

echo "======================================"
echo " Multistage 300-300-400 Sampling"
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
