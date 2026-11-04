#!/bin/bash
# exp03_multi_seed_eval.sh
# ==============================
# 实验目的: 对现有 RDD XL 400K checkpoint 用 3 个不同 seed 生成 50K 样本，
#           计算 FID 的 mean±std，验证评估指标的稳定性。
# 对应论文 Section: Main results - confidence interval / error bars
#
# 配置: 2 cfg × 3 seeds = 6 次采样评估
#   - cfg_scale: 1.0, 1.5
#   - seeds: 0, 42, 123
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp03_multi_seed_eval.sh

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
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/0400000.pt"
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/multi_seed_eval"

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
SEEDS=(0 42 123)
CFG_SCALES=(1.0 1.5)

echo "============================================================"
echo "[exp03] Multi-seed evaluation of RDD XL 400K checkpoint"
echo "  Checkpoint: ${CKPT}"
echo "  Seeds:      ${SEEDS[*]}"
echo "  CFG scales: ${CFG_SCALES[*]}"
echo "  Total runs: $((${#SEEDS[@]} * ${#CFG_SCALES[@]}))"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Run all seed × cfg combinations
# ─────────────────────────────────────────────────────────────────────────────
for CFG in "${CFG_SCALES[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="cfg${CFG}_seed${SEED}"
        OUTPUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
        CONFIG_FILE="${RESULTS_DIR}/config_${RUN_NAME}.jsonl"

        echo "------------------------------------------------------------"
        echo "[exp03] Running: cfg=${CFG}, seed=${SEED}"
        echo "  Output: ${OUTPUT_DIR}"
        echo "------------------------------------------------------------"

        cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":${CFG},"upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
EOF

        # Launch 8-GPU parallel sampling
        for RANK in $(seq 0 7); do
            CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
                --config "$CONFIG_FILE" \
                --rank $RANK --world-size 8 \
                --total-samples 50000 --batch-size 100 \
                --seed $SEED &
        done
        wait
        echo "[exp03] Sampling completed: ${RUN_NAME}"
    done
done

# ─────────────────────────────────────────────────────────────────────────────
# Summarize results: compute mean ± std
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp03] Summary of FID results"
echo "============================================================"

SUMMARY_FILE="${RESULTS_DIR}/summary.txt"
cat /dev/null > "$SUMMARY_FILE"

for CFG in "${CFG_SCALES[@]}"; do
    echo "--- CFG=${CFG} ---" | tee -a "$SUMMARY_FILE"
    FID_VALUES=""
    for SEED in "${SEEDS[@]}"; do
        RUN_NAME="cfg${CFG}_seed${SEED}"
        EVAL_LOG="/home/tiger/multiscale_diffusion/2d/eval_results_unified/${RUN_NAME}.txt"
        if [ -f "$EVAL_LOG" ]; then
            FID=$(grep -i "FID:" "$EVAL_LOG" | head -n1 | awk '{print $NF}')
            echo "  seed=${SEED}: FID=${FID}" | tee -a "$SUMMARY_FILE"
            FID_VALUES="${FID_VALUES} ${FID}"
        else
            echo "  seed=${SEED}: [PENDING - eval log not found]" | tee -a "$SUMMARY_FILE"
        fi
    done

    # Compute mean and std using python
    if [ -n "$FID_VALUES" ]; then
        python3 -c "
import numpy as np
vals = [float(x) for x in '${FID_VALUES}'.split()]
if len(vals) > 0:
    print(f'  Mean FID: {np.mean(vals):.3f} ± {np.std(vals):.3f} (n={len(vals)})')
" | tee -a "$SUMMARY_FILE"
    fi
done

echo ""
echo "[exp03] Full summary saved to: ${SUMMARY_FILE}"
echo "============================================================"
echo "[exp03] All done."
echo "============================================================"
