#!/bin/bash
# DiT-XL/2 500K (Original ImageNet) evaluation: 250-step DDPM, CFG=1.0 and 1.5
# Uses DiT's own sample_fid.py

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/DiT

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

CKPT=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/DiT-XL-2/007-DiT-XL-2/checkpoints/0500000.pt
EVAL_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp_samples_original
TOTAL=50000
BS=64
WS=8

run_experiment() {
    local STEPS=$1
    local CFG=$2
    local TAG=$3

    local OUTDIR="${EVAL_BASE}/${TAG}"
    echo ""
    echo "============================================================"
    echo " Starting: DiT 500k (original), ${STEPS}-step DDPM, CFG=${CFG}"
    echo " Output: ${OUTDIR}"
    echo "============================================================"

    mkdir -p "$OUTDIR"

    for RANK in $(seq 0 $((WS-1))); do
        CUDA_VISIBLE_DEVICES=$RANK python -u sample_fid.py \
            --ckpt $CKPT --model DiT-XL/2 --image-size 256 --num-classes 1000 --vae ema \
            --cfg-scale $CFG --num-sampling-steps $STEPS --sampling-method ddpm \
            --total-samples $TOTAL --batch-size $BS --seed 0 \
            --rank $RANK --world-size $WS \
            --output-dir $OUTDIR &
    done
    wait

    echo "=== Done: ${TAG} ==="
}

run_experiment 250 1.0 "dit500k_250step_ddpm_cfg1.0"
run_experiment 250 1.5 "dit500k_250step_ddpm_cfg1.5"

echo ""
echo "All DiT 500K (original) evaluations done!"
