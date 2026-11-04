#!/bin/bash
# Original DiT 500k: 250-step FID evaluation with 3 sampling methods
# 8-GPU parallel, 50000 samples each, CFG=1.5 + no-CFG (cfg=1.0)

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
STEPS=250
CFG=1.5

mkdir -p $LOGDIR

run_experiment() {
    local STEPS=$1
    local METHOD=$2
    local CFG=$3
    local TAG=$4

    local OUTDIR="${EVAL_BASE}/eval_dit500k_${TAG}"
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

    echo "=== Done: DiT 500k, ${STEPS}-step ${METHOD}, CFG=${CFG} ==="
}

# ============================================================
# 250 steps, CFG=1.5
# ============================================================
run_experiment 250 ddpm 1.5 "250step_ddpm_cfg15"
run_experiment 250 dpm_solver_1 1.5 "250step_dpm1_cfg15"
run_experiment 250 dpm_solver_2 1.5 "250step_dpm2_cfg15"

# ============================================================
# 250 steps, CFG=1.0 (no CFG)
# ============================================================
run_experiment 250 ddpm 1.0 "250step_ddpm_cfg1"
run_experiment 250 dpm_solver_1 1.0 "250step_dpm1_cfg1"
run_experiment 250 dpm_solver_2 1.0 "250step_dpm2_cfg1"

echo ""
echo "=========================================="
echo " All 6 experiments completed!"
echo "=========================================="
echo " Results in: ${EVAL_BASE}/"
echo "   1. DDPM,  CFG=1.5: eval_dit500k_250step_ddpm_cfg15"
echo "   2. DPM1,  CFG=1.5: eval_dit500k_250step_dpm1_cfg15"
echo "   3. DPM2,  CFG=1.5: eval_dit500k_250step_dpm2_cfg15"
echo "   4. DDPM,  CFG=1.0: eval_dit500k_250step_ddpm_cfg1"
echo "   5. DPM1,  CFG=1.0: eval_dit500k_250step_dpm1_cfg1"
echo "   6. DPM2,  CFG=1.0: eval_dit500k_250step_dpm2_cfg1"
echo " Logs in: ${LOGDIR}/"
echo "=========================================="
