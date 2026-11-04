#!/bin/bash
# exp02_dit_iso_flop.sh
# ==============================
# 实验目的: 继续训练 DiT baseline 到 525K 步 (iso-FLOP) 和 750K 步。
# 对应论文 Section: Main comparison - DiT baseline iso-FLOP fairness
#
# 说明:
#   - DiT-XL/2 baseline 使用标准训练脚本 (单分辨率32x32 latent, 无多分辨率)
#   - 需要从已有 DiT checkpoint resume 继续训练
#   - 在 525K 和 750K checkpoint 处做采样评估
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 前置说明:                                                                    │
# │ - DiT 训练脚本位置: 使用 DiT 官方训练代码或项目中的 DiT 训练脚本              │
# │ - DiT checkpoint 路径需要确认，下面使用 placeholder                           │
# │ - 如果项目中没有独立的 DiT 训练脚本，可复用 train_RDD_imagenet256_unified.py  │
# │   并设置 resolutions=32, stage-lengths=1000 (单阶段等同标准 DiT)              │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp02_dit_iso_flop.sh

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
TRAIN_SCRIPT="/home/tiger/multiscale_diffusion/2d/train_RDD_imagenet256_unified.py"
SAMPLE_SCRIPT="/home/tiger/multiscale_diffusion/2d/sample_fid_unified.py"
DATA_PATH="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/data/imagenet/ILSVRC/Data/CLS-LOC/train"
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/DiT-XL-baseline"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"

# TODO: 确认 DiT baseline 已有 checkpoint 路径
# 如果已经有训练到某步的 checkpoint, 填入这里用于 resume
DIT_RESUME_CKPT="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/DiT-XL-baseline/imagenet256-unified/checkpoints/PLACEHOLDER_LATEST.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Training configuration
# DiT baseline = 单分辨率 (32x32 latent), 单阶段 (stage-lengths=1000)
# 这样 unified 模型退化为标准 DiT
# ─────────────────────────────────────────────────────────────────────────────
MODEL="DiT-XL/2"
RESOLUTIONS="32"
STAGE_LENGTHS="1000"
GLOBAL_BATCH_SIZE=256
DIFFUSION_STEPS=1000
EPOCHS=2800  # 足够训练到 750K 步
CKPT_EVERY=5000
VAE="ema"
DATA_SOURCE="imagenet"

echo "============================================================"
echo "[exp02] Training DiT-XL/2 baseline to 525K/750K steps (iso-FLOP)"
echo "  Model:         ${MODEL}"
echo "  Resolutions:   ${RESOLUTIONS} (single-scale = standard DiT)"
echo "  Stage lengths: ${STAGE_LENGTHS}"
echo "  Resume from:   ${DIT_RESUME_CKPT}"
echo "  Output:        ${RESULTS_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Continue training DiT baseline
# ─────────────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=8 "$TRAIN_SCRIPT" \
    --model "$MODEL" \
    --resolutions "$RESOLUTIONS" \
    --stage-lengths "$STAGE_LENGTHS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --diffusion-steps "$DIFFUSION_STEPS" \
    --epochs "$EPOCHS" \
    --ckpt-every "$CKPT_EVERY" \
    --vae "$VAE" \
    --data-source "$DATA_SOURCE" \
    --data-path "$DATA_PATH" \
    --results-dir "$RESULTS_DIR" \
    --resume "$DIT_RESUME_CKPT"

echo "[exp02] Training completed."

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Evaluate at 525K steps
# ─────────────────────────────────────────────────────────────────────────────
CKPT_DIR="${RESULTS_DIR}/imagenet256-unified/checkpoints"
CKPT_525K="${CKPT_DIR}/0525000.pt"
CKPT_750K="${CKPT_DIR}/0750000.pt"

# --- 525K, cfg=1.5 ---
OUTPUT_525K="${RESULTS_DIR}/samples_525K_cfg1.5_ddpm250"
CONFIG_525K="${RESULTS_DIR}/config_525K_cfg1.5.jsonl"

if [ -f "$CKPT_525K" ]; then
    echo "[exp02] Evaluating 525K checkpoint..."
    cat > "$CONFIG_525K" <<EOF
{"model_type":"unified","stage_lengths":"1000","resolutions":"32","model":"DiT-XL/2","ckpt":"${CKPT_525K}","num_steps":"250","sampler":"ddpm","cfg_scale":1.5,"upsample_mode":"fresh","output_dir":"${OUTPUT_525K}"}
EOF

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$CONFIG_525K" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "[exp02] 525K sampling completed."
else
    echo "[exp02] WARNING: ${CKPT_525K} not found, skipping."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Evaluate at 750K steps
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_750K="${RESULTS_DIR}/samples_750K_cfg1.5_ddpm250"
CONFIG_750K="${RESULTS_DIR}/config_750K_cfg1.5.jsonl"

if [ -f "$CKPT_750K" ]; then
    echo "[exp02] Evaluating 750K checkpoint..."
    cat > "$CONFIG_750K" <<EOF
{"model_type":"unified","stage_lengths":"1000","resolutions":"32","model":"DiT-XL/2","ckpt":"${CKPT_750K}","num_steps":"250","sampler":"ddpm","cfg_scale":1.5,"upsample_mode":"fresh","output_dir":"${OUTPUT_750K}"}
EOF

    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$CONFIG_750K" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
    echo "[exp02] 750K sampling completed."
else
    echo "[exp02] WARNING: ${CKPT_750K} not found, skipping."
fi

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Also evaluate both checkpoints with cfg=1.0 for completeness
# ─────────────────────────────────────────────────────────────────────────────
for STEPS in 525000 750000; do
    CKPT_FILE="${CKPT_DIR}/0${STEPS}.pt"
    if [ ! -f "$CKPT_FILE" ]; then
        continue
    fi
    STEP_LABEL=$(echo "$STEPS" | awk '{printf "%dK", $1/1000}')
    OUTPUT_DIR="${RESULTS_DIR}/samples_${STEP_LABEL}_cfg1.0_ddpm250"
    CONFIG_FILE="${RESULTS_DIR}/config_${STEP_LABEL}_cfg1.0.jsonl"

    cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"1000","resolutions":"32","model":"DiT-XL/2","ckpt":"${CKPT_FILE}","num_steps":"250","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
EOF

    echo "[exp02] Sampling ${STEP_LABEL} with cfg=1.0..."
    for RANK in $(seq 0 7); do
        CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
            --config "$CONFIG_FILE" \
            --rank $RANK --world-size 8 \
            --total-samples 50000 --batch-size 100 \
            --seed 0 &
    done
    wait
done

echo "============================================================"
echo "[exp02] All done. Results in: ${RESULTS_DIR}"
echo "============================================================"
