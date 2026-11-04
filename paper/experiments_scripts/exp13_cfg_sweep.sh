#!/bin/bash
# exp13_cfg_sweep.sh
# ==============================
# 实验目的: 完整 CFG sweep — 对DiT和RDD在8个CFG值下评估FID
# 对应 Critique: Round 1 (CFG fairness), Round 32 (CFG curves)
# 对应论文 Section: Appendix / §4.x CFG Analysis
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp13_cfg_sweep.sh

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
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/cfg_sweep"
mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# CFG values to sweep
# ─────────────────────────────────────────────────────────────────────────────
declare -a CFG_VALUES=(1.0 1.25 1.5 1.75 2.0 2.5 3.0 4.0)

echo "============================================================"
echo "[exp13] FID vs CFG Complete Sweep"
echo "  RDD CKPT: ${RDD_CKPT}"
echo "  DiT CKPT: ${DIT_CKPT}"
echo "  CFG values: ${CFG_VALUES[*]}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Part 1: RDD at all CFG values (DDPM 250 steps)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Part 1: RDD CFG sweep"

for CFG in "${CFG_VALUES[@]}"; do
    OUTPUT_DIR="${RESULTS_DIR}/rdd_cfg${CFG}"
    CONFIG_FILE="${RESULTS_DIR}/config_rdd_cfg${CFG}.jsonl"

    echo "  [RDD] cfg=${CFG}"

    cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${RDD_CKPT}","num_steps":"90,80,80","sampler":"ddpm","cfg_scale":"${CFG}","upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
EOF

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$CONFIG_FILE" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "  [RDD] cfg=${CFG} complete."
done

# ─────────────────────────────────────────────────────────────────────────────
# Part 2: DiT at all CFG values (DDPM 250 steps)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo ">>> Part 2: DiT CFG sweep"

for CFG in "${CFG_VALUES[@]}"; do
    OUTPUT_DIR="${RESULTS_DIR}/dit_cfg${CFG}"
    CONFIG_FILE="${RESULTS_DIR}/config_dit_cfg${CFG}.jsonl"

    echo "  [DiT] cfg=${CFG}"

    cat > "$CONFIG_FILE" <<EOF
{"model_type":"dit","model":"DiT-XL/2","ckpt":"${DIT_CKPT}","num_steps":"250","sampler":"ddpm","cfg_scale":"${CFG}","output_dir":"${OUTPUT_DIR}"}
EOF

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$DIT_SAMPLE_SCRIPT" \
            --config "$CONFIG_FILE" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "  [DiT] cfg=${CFG} complete."
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp13] CFG sweep complete!"
echo "Results in: ${RESULTS_DIR}"
echo ""
echo "Expected output structure:"
echo "  ${RESULTS_DIR}/rdd_cfg1.0/  ... rdd_cfg4.0/"
echo "  ${RESULTS_DIR}/dit_cfg1.0/  ... dit_cfg4.0/"
echo ""
echo "Next: compute FID/sFID/IS/P/R for each, then plot FID-vs-CFG curves"
echo "============================================================"
