#!/bin/bash
# exp04_convergence_curve.sh
# ==============================
# 实验目的: 对 RDD 和 DiT 的所有 checkpoint 评估 FID，绘制收敛曲线。
# 对应论文 Section: Convergence analysis / Training efficiency
#
# 配置:
#   - Checkpoints: 50K 到 500K，每 50K 一个（共 10 个）
#   - 采样: 50K samples, cfg=1.0, DDPM 250 步
#   - 模型: RDD-Unified-XL + DiT-XL baseline
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp04_convergence_curve.sh

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

RDD_CKPT_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Unified-XL-300-300-400/imagenet256-unified/checkpoints"
DIT_CKPT_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/DiT-XL-baseline/imagenet256-unified/checkpoints"

RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/convergence_curve"
mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint steps to evaluate
# ─────────────────────────────────────────────────────────────────────────────
STEPS=(50000 100000 150000 200000 250000 300000 350000 400000 450000 500000)

echo "============================================================"
echo "[exp04] Convergence curve evaluation"
echo "  Steps to evaluate: ${STEPS[*]}"
echo "  Sampling config: 50K samples, cfg=1.0, DDPM 250 steps"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Generate JSONL configs for RDD
# ─────────────────────────────────────────────────────────────────────────────
RDD_CONFIG="${RESULTS_DIR}/rdd_convergence.jsonl"
cat /dev/null > "$RDD_CONFIG"

echo "[exp04] Generating RDD sampling configs..."
for STEP in "${STEPS[@]}"; do
    STEP_PADDED=$(printf "%07d" $STEP)
    CKPT_FILE="${RDD_CKPT_DIR}/${STEP_PADDED}.pt"
    STEP_LABEL=$(echo "$STEP" | awk '{printf "%dK", $1/1000}')
    OUTPUT_DIR="${RESULTS_DIR}/rdd_${STEP_LABEL}"

    if [ -f "$CKPT_FILE" ]; then
        echo "{\"model_type\":\"unified\",\"stage_lengths\":\"300,300,400\",\"resolutions\":\"32,16,8\",\"model\":\"DiT-XL/2\",\"ckpt\":\"${CKPT_FILE}\",\"num_steps\":\"80,80,90\",\"sampler\":\"ddpm\",\"cfg_scale\":1.0,\"upsample_mode\":\"fresh\",\"output_dir\":\"${OUTPUT_DIR}\"}" >> "$RDD_CONFIG"
        echo "  [RDD] ${STEP_LABEL}: ${CKPT_FILE}"
    else
        echo "  [RDD] ${STEP_LABEL}: SKIPPED (not found: ${CKPT_FILE})"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Generate JSONL configs for DiT baseline
# ─────────────────────────────────────────────────────────────────────────────
DIT_CONFIG="${RESULTS_DIR}/dit_convergence.jsonl"
cat /dev/null > "$DIT_CONFIG"

echo "[exp04] Generating DiT baseline sampling configs..."
for STEP in "${STEPS[@]}"; do
    STEP_PADDED=$(printf "%07d" $STEP)
    CKPT_FILE="${DIT_CKPT_DIR}/${STEP_PADDED}.pt"
    STEP_LABEL=$(echo "$STEP" | awk '{printf "%dK", $1/1000}')
    OUTPUT_DIR="${RESULTS_DIR}/dit_${STEP_LABEL}"

    if [ -f "$CKPT_FILE" ]; then
        echo "{\"model_type\":\"unified\",\"stage_lengths\":\"1000\",\"resolutions\":\"32\",\"model\":\"DiT-XL/2\",\"ckpt\":\"${CKPT_FILE}\",\"num_steps\":\"250\",\"sampler\":\"ddpm\",\"cfg_scale\":1.0,\"upsample_mode\":\"fresh\",\"output_dir\":\"${OUTPUT_DIR}\"}" >> "$DIT_CONFIG"
        echo "  [DiT] ${STEP_LABEL}: ${CKPT_FILE}"
    else
        echo "  [DiT] ${STEP_LABEL}: SKIPPED (not found: ${CKPT_FILE})"
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# Run RDD evaluation
# ─────────────────────────────────────────────────────────────────────────────
if [ -s "$RDD_CONFIG" ]; then
    echo ""
    echo "[exp04] Running RDD convergence evaluation..."
    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$RDD_CONFIG" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "[exp04] RDD evaluation completed."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run DiT evaluation
# ─────────────────────────────────────────────────────────────────────────────
if [ -s "$DIT_CONFIG" ]; then
    echo ""
    echo "[exp04] Running DiT convergence evaluation..."
    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$DIT_CONFIG" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "[exp04] DiT evaluation completed."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Results summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp04] Collecting FID results into summary CSV"
echo "============================================================"

SUMMARY_CSV="${RESULTS_DIR}/convergence_summary.csv"
echo "model,steps,fid" > "$SUMMARY_CSV"

EVAL_DIR="/home/tiger/multiscale_diffusion/2d/eval_results_unified"

for STEP in "${STEPS[@]}"; do
    STEP_LABEL=$(echo "$STEP" | awk '{printf "%dK", $1/1000}')

    # RDD
    RDD_LOG="${EVAL_DIR}/rdd_${STEP_LABEL}.txt"
    if [ -f "$RDD_LOG" ]; then
        FID=$(grep -i "FID:" "$RDD_LOG" | head -n1 | awk '{print $NF}')
        echo "RDD,${STEP},${FID}" >> "$SUMMARY_CSV"
    fi

    # DiT
    DIT_LOG="${EVAL_DIR}/dit_${STEP_LABEL}.txt"
    if [ -f "$DIT_LOG" ]; then
        FID=$(grep -i "FID:" "$DIT_LOG" | head -n1 | awk '{print $NF}')
        echo "DiT,${STEP},${FID}" >> "$SUMMARY_CSV"
    fi
done

echo "[exp04] Summary CSV: ${SUMMARY_CSV}"
if [ -f "$SUMMARY_CSV" ]; then
    cat "$SUMMARY_CSV"
fi

echo "============================================================"
echo "[exp04] All done. Results in: ${RESULTS_DIR}"
echo "============================================================"
