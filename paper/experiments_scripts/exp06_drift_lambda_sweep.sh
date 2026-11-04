#!/bin/bash
# exp06_drift_lambda_sweep.sh
# ==============================
# 实验目的: 纯采样实验，在推理时将 drift correction 乘以系数 λ，
#           探索 drift 强度对生成质量的影响。
# 对应论文 Section: Ablation - Drift correction scale
#
# 配置:
#   λ = {0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
#   使用已有 400K checkpoint, cfg=1.5, DPM-2 sampler
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 前置修改 (PREREQUISITE CODE CHANGE)                                          │
# │                                                                             │
# │ 需要在 sample_fid_unified.py 中添加 --drift-scale 参数支持:                  │
# │                                                                             │
# │ 1. JSONL配置中添加 "drift_scale" 字段 (默认 1.0):                            │
# │    "drift_scale": 0.5                                                       │
# │                                                                             │
# │ 2. 在采样时将 drift correction 乘以 drift_scale:                             │
# │    drift_correction = drift_scale * original_drift_correction               │
# │                                                                             │
# │ 3. 具体修改位置: gaussian_diffusion_with_drift.py 中的                       │
# │    p_mean_variance 或等效函数，在计算 drift 后乘以 scale 因子。               │
# │    可通过 diffusion 对象的属性传入: diffusion.drift_scale = λ                │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp06_drift_lambda_sweep.sh

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
CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints/0400000.pt"
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/drift_lambda_sweep"

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Lambda values to sweep
# ─────────────────────────────────────────────────────────────────────────────
LAMBDAS=(0 0.25 0.5 0.75 1.0 1.25 1.5)

echo "============================================================"
echo "[exp06] Drift correction λ sweep"
echo "  Checkpoint: ${CKPT}"
echo "  λ values:   ${LAMBDAS[*]}"
echo "  Sampler:    DPM-2, cfg=1.5"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Run each λ configuration
# ─────────────────────────────────────────────────────────────────────────────
for LAMBDA in "${LAMBDAS[@]}"; do
    RUN_NAME="lambda_${LAMBDA}"
    OUTPUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
    CONFIG_FILE="${RESULTS_DIR}/config_${RUN_NAME}.jsonl"

    echo "------------------------------------------------------------"
    echo "[exp06] Running: λ=${LAMBDA}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "------------------------------------------------------------"

    # NOTE: "drift_scale" field requires code modification (see header comment)
    cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"80,80,90","sampler":"dpm2","cfg_scale":1.5,"drift_scale":${LAMBDA},"upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
EOF

    # Launch 8-GPU parallel sampling
    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$CONFIG_FILE" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "[exp06] Completed: λ=${LAMBDA}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp06] Results summary"
echo "============================================================"

EVAL_DIR="/home/tiger/multiscale_diffusion/2d/eval_results_unified"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
echo "drift_lambda,fid" > "$SUMMARY_FILE"

for LAMBDA in "${LAMBDAS[@]}"; do
    RUN_NAME="lambda_${LAMBDA}"
    EVAL_LOG="${EVAL_DIR}/${RUN_NAME}.txt"
    if [ -f "$EVAL_LOG" ]; then
        FID=$(grep -i "FID:" "$EVAL_LOG" | head -n1 | awk '{print $NF}')
        echo "${LAMBDA},${FID}" >> "$SUMMARY_FILE"
        echo "  λ=${LAMBDA} → FID=${FID}"
    else
        echo "  λ=${LAMBDA} → [PENDING]"
    fi
done

echo ""
echo "[exp06] Summary: ${SUMMARY_FILE}"
echo "============================================================"
echo "[exp06] All done."
echo "============================================================"
