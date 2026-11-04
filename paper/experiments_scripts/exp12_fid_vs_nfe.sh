#!/bin/bash
# LEGACY TEMPLATE: remote paths/GPU0-7 are not certified on surfi1. Read PAPER_EXPERIMENT_AUDIT_ZH.md first.
# Scheduled DPM2 steps S are NOT NFE: native3-stage actual calls=2*S-6; CFG branches add cost.
# exp12_fid_vs_nfe.sh
# ==============================
# 实验目的: FID vs NFE Pareto Curve — 不同采样步数下DiT和RDD的质量-速度权衡
# 对应 Critique: Round 31 (efficiency), Round 33 (noise schedule)
# 对应论文 Section: §4.x "Quality-Speed Pareto Analysis"
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp12_fid_vs_nfe.sh

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
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/pareto_nfe"
mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# NFE configurations
# For RDD: stage steps are distributed proportionally (36%/32%/32% of total NFE)
# All use DPM-Solver-2 sampler and cfg=1.5
# ─────────────────────────────────────────────────────────────────────────────
declare -a NFES=(25 50 65 100 125 250)

# Precomputed per-stage steps (coarse/mid/fine) for RDD — must sum to total NFE
# Using ceil-floor rounding to ensure sum = NFE
declare -a RDD_STEPS=(
    "9,8,8"       # NFE=25: 9+8+8=25
    "18,16,16"    # NFE=50: 18+16+16=50
    "24,21,20"    # NFE=65: 24+21+20=65
    "36,32,32"    # NFE=100: 36+32+32=100
    "45,40,40"    # NFE=125: 45+40+40=125
    "90,80,80"    # NFE=250: 90+80+80=250
)

echo "============================================================"
echo "[exp12] FID vs NFE Pareto Curve"
echo "  RDD CKPT: ${RDD_CKPT}"
echo "  DiT CKPT: ${DIT_CKPT}"
echo "  NFEs: ${NFES[*]}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: RDD sampling at various NFE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Part 1: RDD at various NFE"

for idx in "${!NFES[@]}"; do
    NFE=${NFES[$idx]}
    STEPS=${RDD_STEPS[$idx]}
    OUTPUT_DIR="${RESULTS_DIR}/rdd_nfe${NFE}"
    CONFIG_FILE="${RESULTS_DIR}/config_rdd_nfe${NFE}.jsonl"

    echo "  [RDD] NFE=${NFE}, steps=${STEPS}"

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
# Part 2: DiT sampling at various NFE
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Part 2: DiT at various NFE"

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
echo "[exp12] All NFE experiments complete!"
echo "Results in: ${RESULTS_DIR}"
echo ""
echo "Next step: run FID evaluation on each output directory"
echo "  for dir in ${RESULTS_DIR}/*/; do"
echo "    python evaluator.py \$REFERENCE_BATCH \$dir/samples.npz"
echo "  done"
echo "============================================================"
