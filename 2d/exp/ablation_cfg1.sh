#!/bin/bash
# Ablation study with CFG=1.0 (unconditional, proper ablation setting)
# Unified XL 400K: DPM-Solver-2 vs DDPM step sweep + Stage split sensitivity
# Multistage 400K: DPM-Solver-2 step sweep
# 15 experiments total, ~2h each with 8 GPUs = ~30h total (sequential)

set +e

source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/2d

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

echo "Starting CFG=1.0 ablation experiments (15 total)..."

torchrun --nproc_per_node=8 --master_port=29501 \
    sample_fid_unified.py \
    --config /home/tiger/multiscale_diffusion/2d/exp/ablation_cfg1.jsonl \
    --total-samples 50000 \
    --batch-size 64 \
    --seed 0

echo ""
echo "All CFG=1.0 ablation experiments done!"
