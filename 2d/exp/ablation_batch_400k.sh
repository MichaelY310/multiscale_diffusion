#!/bin/bash
# Ablation experiments batch:
#   1. Stage step allocation (multistage & unified, 4+4=8 experiments)
#   2. Bilinear upsample vs noise_filling (2 experiments)
#   3. FID vs Total Steps curve (multistage, 5 experiments)
# Total: 15 experiments, all cfg=1.5 (except bilinear which is cfg=1.0)
# Timing is automatically logged by sample_fid_unified.py per batch and per experiment.

NUM_GPUS=8
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SAMPLE_SCRIPT="${SCRIPT_DIR}/../sample_fid_unified.py"
CONFIG="${SCRIPT_DIR}/ablation_batch_400k.jsonl"

echo "======================================"
echo " Ablation Batch Experiments (400K step)"
echo " Config: ${CONFIG}"
echo " Experiments: 15 total"
echo "   - Step allocation (ms): 4 configs"
echo "   - Step allocation (uni): 4 configs"
echo "   - Bilinear upsample: 2 configs"
echo "   - FID vs Total Steps: 5 configs"
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

echo "All ablation experiments done!"
