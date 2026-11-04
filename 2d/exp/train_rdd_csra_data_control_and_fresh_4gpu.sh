#!/usr/bin/env bash
# Match the 4K-data arm's initialization, optimizer and validation corpus.
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
INITIAL=checkpoints/RDD-CSRA-output-continue2k-L400K-20260912/best.pt
OUTPUT=checkpoints/RDD-CSRA-output-data512-control-L400K-20260912
EVAL=eval_results/csra_output_data512_control_best_20260912
FRESH=eval_results/csra_output_data4k_fresh_20260912
LARGE=checkpoints/RDD-CSRA-output-data4k-L400K-20260912/best.pt
[[ -f "$INITIAL" && -f "$LARGE" ]] || exit 1
for path in "$OUTPUT" "$EVAL" "$FRESH"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite $path"; exit 1; }
done
mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
[[ ${#free_mib[@]} == 4 ]] || exit 1
for free in "${free_mib[@]}"; do (( free >= 12000 )) || exit 1; done
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/train_csra_rollout_output.py \
  --base-ckpt checkpoints/RDD-Unified-L/0400000.pt --init-adapter "$INITIAL" \
  --data-root eval_results/csra_rollout_data_20260912 \
  --validation-data-root eval_results/csra_rollout_data4k_20260912 \
  --output-dir "$OUTPUT" --max-steps 800 --global-batch-size 256 \
  --learning-rate 1e-4 --variance-weight 0.05 --validate-every 100 \
  --checkpoint-every 200 --seed 20274000
[[ -f "$OUTPUT/complete.json" && -f "$OUTPUT/best.pt" ]] || exit 1
ADAPTER_CKPT="$ROOT/$OUTPUT/best.pt" OUT_ROOT="$ROOT/$EVAL" \
  ADAPTER_PRECISION=fp32 SESSION_PREFIX=rdd_csra_output_data512_control_best \
  bash 2d/exp/eval_rdd_csra_onpolicy_late6_3seed.sh
# Control previews use 0/1/2; fresh-seed diagnostics run serially on GPU3.
# No new seeds are used to choose checkpoints. These are quality-only samples.
for seed in 20275001 20275002 20275003 20275004; do
  mkdir -p "$FRESH/seed_$seed"
  CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u 2d/attention_analysis/sample_csra_adapter_preview.py \
    --family rdd --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
    --adapter-ckpt "$LARGE" \
    --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
    --output-dir "$FRESH/seed_$seed" --seed "$seed" --batch-size 4 \
    --rdd-steps 80,80,90 --benchmark-repeats 0 --adapter-precision fp32 \
    --device cuda:0 2>&1 | tee "$FRESH/seed_$seed/run.log"
done
echo 'Matched-data control and fresh-seed preview dispatch completed.'
