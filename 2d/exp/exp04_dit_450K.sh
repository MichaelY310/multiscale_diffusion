#!/bin/bash
# dit_450K: 采样 + FID 计算

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/DiT

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

EVAL_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp04_convergence
CKPT_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/DiT-XL-2
TOTAL=50000
BS=256
WS=8

REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
PNG_TO_NPZ="/home/tiger/multiscale_diffusion/2d/png_to_npz.py"
LOG_DIR="/home/tiger/multiscale_diffusion/2d/exp/eval_results_exp04"
mkdir -p "$LOG_DIR"

TAG="dit_450K"
CKPT="${CKPT_BASE}/007-DiT-XL-2/checkpoints/0450000.pt"
OUTDIR="${EVAL_BASE}/${TAG}"

echo ""
echo "============================================================"
echo " Sampling: ${TAG}, DDPM 250step, CFG=1.0"
echo " Checkpoint: ${CKPT}"
echo " Output: ${OUTDIR}"
echo "============================================================"

mkdir -p "$OUTDIR"

for RANK in $(seq 0 $((WS-1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u sample_fid.py \
        --ckpt $CKPT --model DiT-XL/2 --image-size 256 --num-classes 1000 --vae ema \
        --cfg-scale 1.0 --num-sampling-steps 250 --sampling-method ddpm \
        --total-samples $TOTAL --batch-size $BS --seed 0 \
        --rank $RANK --world-size $WS \
        --output-dir $OUTDIR &
done
wait
echo "=== Sampling done: ${TAG} ==="

# FID 计算
LOG_FILE="${LOG_DIR}/${TAG}.txt"
LOCAL_NPZ="/tmp/${TAG}_samples.npz"

echo "  [1/2] Packing PNG -> NPZ..."
python "$PNG_TO_NPZ" "$OUTDIR" --num-samples 50000 --output "$LOCAL_NPZ"

echo "  [2/2] Computing FID..."
python -u "$EVALUATOR" "$REF_NPZ" "$LOCAL_NPZ" 2>&1 | tee "$LOG_FILE"
echo "  Result:"
grep "^FID\|^sFID\|^Precision\|^Recall" "$LOG_FILE"
rm -f "$LOCAL_NPZ"

echo ""
echo "All done!"
