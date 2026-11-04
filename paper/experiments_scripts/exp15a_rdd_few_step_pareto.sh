#!/bin/bash
# exp15a_rdd_few_step_pareto.sh
# ==============================
# 实验目的: RDD在极低NFE (4-250步) 下的FID Pareto曲线
# 证明RDD即使在4-16步也能维持合理质量（为蒸馏提供ceiling参考）
# 对应 Critique: Round 17/20/33, 组合加速论证
# 对应论文 Section: §4.x "Composability with Step-Reduction Methods"
#
# 注意：这个实验和exp12有重叠但更强调极低NFE
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp15a_rdd_few_step_pareto.sh

set -e

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface"
export TMPDIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/tmp"
mkdir -p "$TMPDIR"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_SCRIPT="/home/tiger/multiscale_diffusion/2d/sample_fid_unified.py"
DIT_SAMPLE_SCRIPT="/home/tiger/multiscale_diffusion/2d/sample_fid_dit.py"
RDD_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/0400000.pt"
DIT_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/DiT_baseline/DiT-XL-2-ImageNet-256/checkpoints/0500000.pt"
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/few_step_pareto"
mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# NFE configurations (极低步数重点)
# RDD steps: allocated as (36%, 32%, 32%) across 3 stages, rounded
# ─────────────────────────────────────────────────────────────────────────────
declare -a NFES=(4 6 8 12 16 25 50 100 250)

# Per-stage step allocation (coarse/mid/fine, sum=total NFE)
declare -a RDD_STEPS=(
    "2,1,1"        # NFE=4
    "2,2,2"        # NFE=6
    "3,3,2"        # NFE=8
    "5,4,3"        # NFE=12
    "6,5,5"        # NFE=16
    "9,8,8"        # NFE=25
    "18,16,16"     # NFE=50
    "36,32,32"     # NFE=100
    "90,80,80"     # NFE=250
)

echo "============================================================"
echo "[exp15a] RDD Few-Step Pareto Frontier"
echo "  RDD CKPT: ${RDD_CKPT}"
echo "  DiT CKPT: ${DIT_CKPT}"
echo "  NFEs: ${NFES[*]}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: RDD at all NFE (DPM-Solver-2)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Part 1: RDD Unified XL at all NFE"

for idx in "${!NFES[@]}"; do
    NFE=${NFES[$idx]}
    STEPS=${RDD_STEPS[$idx]}
    OUTPUT_DIR="${RESULTS_DIR}/rdd_nfe${NFE}"
    CONFIG_FILE="${RESULTS_DIR}/config_rdd_nfe${NFE}.jsonl"

    echo "  [RDD] NFE=${NFE}, steps_per_stage=${STEPS}"

    cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${RDD_CKPT}","num_steps":"${STEPS}","sampler":"dpm_solver","cfg_scale":"1.5","upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
EOF

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$CONFIG_FILE" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "  [RDD] NFE=${NFE} complete."
done

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: DiT at matching NFE (for comparison)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Part 2: DiT at matching NFE"

for NFE in "${NFES[@]}"; do
    OUTPUT_DIR="${RESULTS_DIR}/dit_nfe${NFE}"
    CONFIG_FILE="${RESULTS_DIR}/config_dit_nfe${NFE}.jsonl"

    echo "  [DiT] NFE=${NFE}"

    cat > "$CONFIG_FILE" <<EOF
{"model_type":"dit","model":"DiT-XL/2","ckpt":"${DIT_CKPT}","num_steps":"${NFE}","sampler":"dpm_solver","cfg_scale":"1.5","output_dir":"${OUTPUT_DIR}"}
EOF

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$DIT_SAMPLE_SCRIPT" \
            --config "$CONFIG_FILE" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "  [DiT] NFE=${NFE} complete."
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp15a] Few-step Pareto experiment complete!"
echo ""
echo "Expected key findings:"
echo "  - At NFE=25: RDD should match DiT at NFE=250 (FID within ±1)"
echo "  - At NFE=8-16: RDD should be significantly better than DiT at same NFE"
echo "  - At NFE=4: RDD FID should be <15 (viable for distillation starting point)"
echo ""
echo "Results in: ${RESULTS_DIR}"
echo "Next: compute FID for each, then plot Pareto curve (FID vs NFE)"
echo "============================================================"
