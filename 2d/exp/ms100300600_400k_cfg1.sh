#!/bin/bash
# RDD-Multistage-100-300-600 400K: cfg=1.0, ddpm 80+80+90, noise_filling
set +e
source /home/tiger/multiscale_diffusion/2d/2d/bin/activate
cd /home/tiger/multiscale_diffusion/2d

export HF_HUB_OFFLINE=1
unset no_proxy NO_PROXY http_proxy https_proxy HTTP_PROXY HTTPS_PROXY HF_ENDPOINT
export HF_HOME=/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/.cache/huggingface

NUM_GPUS=8
SAMPLE_SCRIPT="$(cd "$(dirname "$0")" && pwd)/../sample_fid_unified.py"

REF_NPZ="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/VIRTUAL_imagenet256_labeled.npz"
EVALUATOR="/home/tiger/multiscale_diffusion/guided-diffusion/evaluations/evaluator.py"
PNG_TO_NPZ="/home/tiger/multiscale_diffusion/2d/png_to_npz.py"
RESULT_DIR="/home/tiger/multiscale_diffusion/2d/eval_results_unified"
mkdir -p "$RESULT_DIR"

CONFIG="/tmp/ms100300600_400k_cfg1.jsonl"
cat > "$CONFIG" << 'EOF'
{"model_type":"multistage","stage_lengths":"100,300,600","resolutions":"32,16,8","model_8":"DiT-B/2","model_16":"DiT-L/2","model_32":"DiT-XL/2","ckpt_dir":"/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/ImageNet_original/RDD-Multistage-100-300-600/003-imagenet256-multiscale","ckpt_step":"0400000","num_steps":"80,80,90","sampler":"ddpm","cfg_scale":1.0,"upsample_mode":"fresh","output_dir":"/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/eval_mse/ms100300600_400k_808090_ddpm_cfg1_noisefilling"}
EOF

TAG="ms100300600_400k_808090_ddpm_cfg1_noisefilling"
OUTDIR="/mnt/bn/yangyuan-search-agent-rl-hl/mlx/users/yangyuan.335/RDD/eval_mse/${TAG}"

echo "======================================"
echo " RDD-Multistage-100-300-600 400K cfg=1.0"
echo "======================================"

# 采样
for RANK in $(seq 0 $((NUM_GPUS-1))); do
    CUDA_VISIBLE_DEVICES=$RANK python -u "${SAMPLE_SCRIPT}" \
        --config "${CONFIG}" \
        --rank $RANK --world-size $NUM_GPUS \
        --total-samples 50000 --batch-size 256 \
        --seed 0 &
done
wait

echo "=== Sampling done! Computing FID... ==="

FILE_COUNT=$(find "$OUTDIR" -maxdepth 1 -name "*.png" -type f | wc -l)
echo "Found $FILE_COUNT PNG files"
if [ "$FILE_COUNT" -lt 49900 ]; then
    echo "Not enough files, exiting"
    exit 1
fi

LOCAL_NPZ="/tmp/${TAG}_samples.npz"
echo "[1/2] Packing PNG -> NPZ..."
python "$PNG_TO_NPZ" "$OUTDIR" --num-samples 50000 --output "$LOCAL_NPZ"

echo "[2/2] Computing FID..."
python -u "$EVALUATOR" "$REF_NPZ" "$LOCAL_NPZ" 2>&1 | tee "$RESULT_DIR/${TAG}.txt"
echo "Done!"
grep "^FID\|^sFID\|^Precision\|^Recall" "$RESULT_DIR/${TAG}.txt"
rm -f "$LOCAL_NPZ"
