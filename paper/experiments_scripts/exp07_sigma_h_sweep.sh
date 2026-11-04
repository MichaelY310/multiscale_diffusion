#!/bin/bash
# exp07_sigma_h_sweep.sh
# ==============================
# 实验目的: 纯采样实验，在 noise filling upsample 时使用不同 σ_h。
#           σ_h 控制 null-space 高频分量的先验强度。
# 对应论文 Section: Ablation - Noise filling σ_h
#
# 配置:
#   σ_h = {0, 0.05, 0.1, 0.2, 0.5, 1.0}
#   使用已有 400K checkpoint, cfg=1.5, DDPM 250步 (80,80,90)
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 前置修改 (PREREQUISITE CODE CHANGE)                                          │
# │                                                                             │
# │ 需要在 sample_fid_unified.py 中添加 --sigma-h 参数支持:                      │
# │                                                                             │
# │ 1. JSONL配置中添加 "sigma_h" 字段 (默认 0，即标准 noise filling):            │
# │    "sigma_h": 0.1                                                           │
# │                                                                             │
# │ 2. 修改 _noise_filling_upsample() 函数:                                     │
# │    当 σ_h > 0 时，null-space 分量使用独立噪声:                               │
# │      x_T = sqrt(α) * U(x0) + sqrt(1-α) * (UD·ε + σ_h * P·ε')             │
# │    其中 P = I - UD 是 null-space 投影，ε' 是独立噪声。                       │
# │    当 σ_h = 0 时，退化为标准 q_sample (等同当前实现)。                        │
# │    当 σ_h = 1 时，等同标准 noise filling (当前实现)。                         │
# │                                                                             │
# │ 3. 通过 exp config 传入: run_experiment() 读取 sigma_h 并传给                │
# │    _do_upsample()。                                                          │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp07_sigma_h_sweep.sh

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
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/sigma_h_sweep"

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# σ_h values to sweep
# ─────────────────────────────────────────────────────────────────────────────
SIGMA_H_VALUES=(0 0.05 0.1 0.2 0.5 1.0)

echo "============================================================"
echo "[exp07] Noise filling σ_h sweep"
echo "  Checkpoint: ${CKPT}"
echo "  σ_h values: ${SIGMA_H_VALUES[*]}"
echo "  Sampler:    DDPM 250步 (80,80,90), cfg=1.5"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Run each σ_h configuration
# ─────────────────────────────────────────────────────────────────────────────
for SIGMA_H in "${SIGMA_H_VALUES[@]}"; do
    RUN_NAME="sigma_h_${SIGMA_H}"
    OUTPUT_DIR="${RESULTS_DIR}/${RUN_NAME}"
    CONFIG_FILE="${RESULTS_DIR}/config_${RUN_NAME}.jsonl"

    echo "------------------------------------------------------------"
    echo "[exp07] Running: σ_h=${SIGMA_H}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "------------------------------------------------------------"

    # NOTE: "sigma_h" field requires code modification (see header comment)
    cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.5,"sigma_h":${SIGMA_H},"upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
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
    echo "[exp07] Completed: σ_h=${SIGMA_H}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp07] Results summary"
echo "============================================================"

EVAL_DIR="/home/tiger/multiscale_diffusion/2d/eval_results_unified"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
echo "sigma_h,fid" > "$SUMMARY_FILE"

for SIGMA_H in "${SIGMA_H_VALUES[@]}"; do
    RUN_NAME="sigma_h_${SIGMA_H}"
    EVAL_LOG="${EVAL_DIR}/${RUN_NAME}.txt"
    if [ -f "$EVAL_LOG" ]; then
        FID=$(grep -i "FID:" "$EVAL_LOG" | head -n1 | awk '{print $NF}')
        echo "${SIGMA_H},${FID}" >> "$SUMMARY_FILE"
        echo "  σ_h=${SIGMA_H} → FID=${FID}"
    else
        echo "  σ_h=${SIGMA_H} → [PENDING]"
    fi
done

echo ""
echo "[exp07] Summary: ${SUMMARY_FILE}"
echo "============================================================"
echo "[exp07] All done."
echo "============================================================"
