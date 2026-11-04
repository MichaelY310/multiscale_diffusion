#!/bin/bash
# exp05_per_stage_cfg.sh
# ==============================
# 实验目的: 纯采样实验，在推理时对每个 stage 使用不同的 CFG scale。
# 对应论文 Section: Ablation - Per-stage classifier-free guidance
#
# 配置 grid (coarse/mid/fine = stage1/stage2/stage3):
#   [1.5/1.5/1.5]    -- baseline (uniform)
#   [2.0/1.5/1.25]   -- decreasing
#   [2.0/1.5/1.0]    -- strong coarse, no cfg fine
#   [1.5/1.25/1.0]   -- mild decreasing
#   [2.5/1.5/1.0]    -- aggressive coarse
#
# 使用已有 RDD XL 400K checkpoint
#
# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ 前置修改 (PREREQUISITE CODE CHANGE)                                          │
# │                                                                             │
# │ sample_fid_unified.py 目前只支持单个 cfg_scale 参数。                         │
# │ 需要修改以支持 per-stage cfg:                                                │
# │                                                                             │
# │ 1. JSONL配置中将 "cfg_scale" 改为支持逗号分隔格式:                            │
# │    "cfg_scale": "2.0,1.5,1.25"  (按 coarse→fine 顺序)                       │
# │                                                                             │
# │ 2. 在 _sample_batch_unified() 中解析 per-stage cfg_scale，                   │
# │    每个 stage 使用对应的 cfg_scale 构建 model_kwargs。                        │
# │                                                                             │
# │ 3. 如果 cfg_scale 是单个数值，则所有 stage 使用相同值（向后兼容）。            │
# └─────────────────────────────────────────────────────────────────────────────┘
#
# 用法:
#   source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
#   bash exp05_per_stage_cfg.sh

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
RESULTS_DIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/per_stage_cfg"

mkdir -p "$RESULTS_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Per-stage CFG configurations
# Format: "coarse_cfg,mid_cfg,fine_cfg" (sampling order: coarse→fine)
# ─────────────────────────────────────────────────────────────────────────────
declare -a CFG_CONFIGS=(
    "1.5,1.5,1.5"
    "2.0,1.5,1.25"
    "2.0,1.5,1.0"
    "1.5,1.25,1.0"
    "2.5,1.5,1.0"
)

echo "============================================================"
echo "[exp05] Per-stage CFG sweep"
echo "  Checkpoint: ${CKPT}"
echo "  Configs:    ${#CFG_CONFIGS[@]} combinations"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Generate JSONL and run each configuration
# ─────────────────────────────────────────────────────────────────────────────
for CFG_STR in "${CFG_CONFIGS[@]}"; do
    # Create a label by replacing commas with underscores
    CFG_LABEL=$(echo "$CFG_STR" | tr ',' '_')
    OUTPUT_DIR="${RESULTS_DIR}/cfg_${CFG_LABEL}"
    CONFIG_FILE="${RESULTS_DIR}/config_cfg_${CFG_LABEL}.jsonl"

    echo "------------------------------------------------------------"
    echo "[exp05] Running per-stage cfg=${CFG_STR}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "------------------------------------------------------------"

    # NOTE: cfg_scale uses comma-separated format for per-stage cfg
    # This requires the code modification described above
    cat > "$CONFIG_FILE" <<EOF
{"model_type":"unified","stage_lengths":"300,300,400","resolutions":"32,16,8","model":"DiT-XL/2","ckpt":"${CKPT}","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":"${CFG_STR}","upsample_mode":"fresh","output_dir":"${OUTPUT_DIR}"}
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
    echo "[exp05] Completed: cfg=${CFG_STR}"
done

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "[exp05] Results summary"
echo "============================================================"

EVAL_DIR="/home/tiger/multiscale_diffusion/2d/eval_results_unified"
SUMMARY_FILE="${RESULTS_DIR}/summary.csv"
echo "cfg_coarse,cfg_mid,cfg_fine,fid" > "$SUMMARY_FILE"

for CFG_STR in "${CFG_CONFIGS[@]}"; do
    CFG_LABEL=$(echo "$CFG_STR" | tr ',' '_')
    EVAL_LOG="${EVAL_DIR}/cfg_${CFG_LABEL}.txt"
    if [ -f "$EVAL_LOG" ]; then
        FID=$(grep -i "FID:" "$EVAL_LOG" | head -n1 | awk '{print $NF}')
        echo "${CFG_STR},${FID}" >> "$SUMMARY_FILE"
        echo "  cfg=${CFG_STR} → FID=${FID}"
    else
        echo "  cfg=${CFG_STR} → [PENDING]"
    fi
done

echo ""
echo "[exp05] Summary: ${SUMMARY_FILE}"
echo "============================================================"
echo "[exp05] All done."
echo "============================================================"
