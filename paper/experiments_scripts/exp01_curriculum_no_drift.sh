#!/bin/bash
# LEGACY TEMPLATE: remote paths/GPU0-7 are not certified on surfi1. Read PAPER_EXPERIMENT_AUDIT_ZH.md first.
# Existing trainer already implements --no-drift; prerequisite comments below are historical.
# exp01_curriculum_no_drift.sh
# ==============================
# 实验目的: 训练一个 d^c=0 (无drift correction) 的模型，用于验证 drift correction 的作用。
# 对应论文 Section: Ablation - Drift Correction
#
# 说明: 仍然做多分辨率 curriculum 训练 (resolutions 32,16,8)，但在扩散过程中
#       不使用 drift correction (即 d^c=0)。这需要在训练代码中添加 --no-drift flag。
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 前置修改 (PREREQUISITE CODE CHANGE)                                          │
# │                                                                             │
# │ 需要在 train_RDD_imagenet256_unified.py 中添加 --no-drift 参数:              │
# │                                                                             │
# │   parser.add_argument("--no-drift", action="store_true",                    │
# │       help="Disable drift correction (d^c=0). Model still trains with "     │
# │            "multi-resolution curriculum but without drift.")                 │
# │                                                                             │
# │ 在 create_diffusion_with_drift() 调用处，当 args.no_drift=True 时，          │
# │ 传入 drift_scale=0 或等效参数使 drift correction 全部置零。                    │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp01_curriculum_no_drift.sh

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
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/RDD-NoDrift-XL-300-300-400"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"

# ─────────────────────────────────────────────────────────────────────────────
# Training configuration (same as RDD XL, plus --no-drift)
# ─────────────────────────────────────────────────────────────────────────────
MODEL="DiT-XL/2"
RESOLUTIONS="32,16,8"
STAGE_LENGTHS="300,300,400"
GLOBAL_BATCH_SIZE=256
DIFFUSION_STEPS=1000
EPOCHS=1400
CKPT_EVERY=5000
VAE="ema"
DATA_SOURCE="imagenet"

echo "============================================================"
echo "[exp01] Training RDD XL with NO DRIFT (d^c=0)"
echo "  Model:         ${MODEL}"
echo "  Resolutions:   ${RESOLUTIONS}"
echo "  Stage lengths: ${STAGE_LENGTHS}"
echo "  No-drift:      YES"
echo "  Output:        ${RESULTS_DIR}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Training
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
    --no-drift

echo "[exp01] Training completed."

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Find the final checkpoint
# ─────────────────────────────────────────────────────────────────────────────
CKPT_DIR="${RESULTS_DIR}/imagenet256-unified/checkpoints"
FINAL_CKPT=$(ls -t "${CKPT_DIR}"/*.pt 2>/dev/null | head -n1)
if [ -z "$FINAL_CKPT" ]; then
    echo "[exp01] ERROR: No checkpoint found in ${CKPT_DIR}"
    exit 1
fi
echo "[exp01] Using checkpoint: ${FINAL_CKPT}"

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Sampling & Evaluation - CFG=1.0
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_CFG10="${RESULTS_DIR}/samples_cfg1.0_ddpm250"
CONFIG_CFG10="${RESULTS_DIR}/config_cfg1.0.jsonl"

cat > "$CONFIG_CFG10" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${FINAL_CKPT}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"${OUTPUT_CFG10}"}
EOF

echo "[exp01] Sampling with cfg=1.0 ..."
for RANK in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
        --config "$CONFIG_CFG10" \
        --rank $RANK --world-size 8 \
        --total-samples 50000 --batch-size 100 \
        --seed 0 &
done
wait
echo "[exp01] Sampling cfg=1.0 completed."

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Sampling & Evaluation - CFG=1.5
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_CFG15="${RESULTS_DIR}/samples_cfg1.5_ddpm250"
CONFIG_CFG15="${RESULTS_DIR}/config_cfg1.5.jsonl"

cat > "$CONFIG_CFG15" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${FINAL_CKPT}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.5,"upsample_mode":"fresh","output_dir":"${OUTPUT_CFG15}"}
EOF

echo "[exp01] Sampling with cfg=1.5 ..."
for RANK in $(seq 0 7); do
    CUDA_VISIBLE_DEVICES=$RANK python -u "$SAMPLE_SCRIPT" \
        --config "$CONFIG_CFG15" \
        --rank $RANK --world-size 8 \
        --total-samples 50000 --batch-size 100 \
        --seed 0 &
done
wait
echo "[exp01] Sampling cfg=1.5 completed."

echo "============================================================"
echo "[exp01] All done. Results in: ${RESULTS_DIR}"
echo "============================================================"
