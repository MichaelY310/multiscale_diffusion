#!/bin/bash
# exp04 机器B: DiT convergence curve (cfg=1.0, ddpm 250step)
# 使用 DiT 原版 sample_fid.py（非 unified 版本）
# 优先: 100K, 200K, 300K; 补充: 50K, 150K, 250K, 350K, 450K
# 注: DiT 400K 和 500K 已有 50K 采样

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/DiT

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

EVAL_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/exp04_convergence
TOTAL=50000
BS=256
WS=8

run_dit() {
    local CKPT=$1
    local TAG=$2

    local OUTDIR="${EVAL_BASE}/${TAG}"

    # 跳过已完成
    local EXISTING=$(find "$OUTDIR" -name "*.png" 2>/dev/null | wc -l)
    if [ "$EXISTING" -ge "$TOTAL" ]; then
        echo "=== SKIP ${TAG}: already has ${EXISTING} images ==="
        return
    fi

    echo ""
    echo "============================================================"
    echo " Starting: ${TAG}, DDPM 250step, CFG=1.0"
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
    echo "=== Done: ${TAG} ==="
}

CKPT_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/DiT-XL-2

# PHASE 1: 优先
run_dit "${CKPT_BASE}/004-DiT-XL-2/checkpoints/0100000.pt" "dit_100K"
run_dit "${CKPT_BASE}/005-DiT-XL-2/checkpoints/0200000.pt" "dit_200K"
run_dit "${CKPT_BASE}/006-DiT-XL-2/checkpoints/0300000.pt" "dit_300K"

# PHASE 2: 补充
run_dit "${CKPT_BASE}/004-DiT-XL-2/checkpoints/0050000.pt" "dit_50K"
run_dit "${CKPT_BASE}/004-DiT-XL-2/checkpoints/0150000.pt" "dit_150K"
run_dit "${CKPT_BASE}/006-DiT-XL-2/checkpoints/0250000.pt" "dit_250K"
run_dit "${CKPT_BASE}/006-DiT-XL-2/checkpoints/0350000.pt" "dit_350K"
run_dit "${CKPT_BASE}/007-DiT-XL-2/checkpoints/0450000.pt" "dit_450K"

echo ""
echo "All DiT convergence sampling done!"
