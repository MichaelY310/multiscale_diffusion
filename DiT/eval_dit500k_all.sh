#!/bin/bash
# Original DiT 500k: Remaining 3 experiments
# 1. 50 steps, DPM2, cfg=1.0 (resume from 17188/50000)
# 2. 100 steps, DDPM, cfg=1.0
# 3. 100 steps, DPM2, cfg=1.0

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/DiT

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

CKPT=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/DiT_ImageNet256/007-DiT-XL-2/checkpoints/0500000.pt
EVAL_BASE=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/eval
LOGDIR=/home/tiger/multiscale_diffusion/DiT/logs
TOTAL=50000
BS=64
WS=8

mkdir -p $LOGDIR

run_experiment() {
    local STEPS=$1
    local METHOD=$2
    local CFG=$3
    local TAG=$4

    local OUTDIR="${EVAL_BASE}/${TAG}"
    echo ""
    echo "============================================================"
    echo " Starting: DiT 500k, ${STEPS}-step ${METHOD}, CFG=${CFG}"
    echo " Output: ${OUTDIR}"
    echo "============================================================"

    mkdir -p "$OUTDIR"

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u sample_fid.py \
            --ckpt $CKPT --model DiT-XL/2 --image-size 256 --num-classes 1000 --vae ema \
            --cfg-scale $CFG --num-sampling-steps $STEPS --sampling-method $METHOD \
            --total-samples $TOTAL --batch-size $BS --seed 0 \
            --rank $RANK --world-size $WS \
            --output-dir $OUTDIR > ${LOGDIR}/${TAG}_rank${RANK}.log 2>&1 &
    done
    wait

    echo "=== Done: ${TAG} ==="
}

# 1. 50 steps, DPM2, cfg=1.0 (resume - has 17188 in old dir name with bug)
run_experiment 50 dpm_solver_2 1.0 "eval_dit500k_50step_50step_dpm2_cfg1"

# 2. 100 steps, DDPM, cfg=1.0
run_experiment 100 ddpm 1.0 "eval_dit500k_100step_ddpm_cfg1"

# 3. 100 steps, DPM2, cfg=1.0
run_experiment 100 dpm_solver_2 1.0 "eval_dit500k_100step_dpm2_cfg1"

echo ""
echo "=========================================="
echo " All 3 remaining experiments completed!"
echo "=========================================="
echo " Results in: ${EVAL_BASE}/"
echo "   1. eval_dit500k_50step_dpm2_cfg1     (50-step DPM2, cfg=1.0)"
echo "   2. eval_dit500k_100step_ddpm_cfg1    (100-step DDPM, cfg=1.0)"
echo "   3. eval_dit500k_100step_dpm2_cfg1    (100-step DPM2, cfg=1.0)"
echo " Logs in: ${LOGDIR}/"
echo "=========================================="
