#!/usr/bin/env bash
# Extend the still-improving 4K-data arm; same data/optimizer, no backbone updates.
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
INITIAL=checkpoints/RDD-CSRA-output-data4k-L400K-20260912/0000800.pt
OUTPUT=checkpoints/RDD-CSRA-output-data4k-continue3200-L400K-20260912
EVAL=eval_results/csra_output_data4k_continue3200_best_20260912
FRESH=eval_results/csra_output_data4k_continue3200_fresh_20260912
[[ -f "$INITIAL" ]] || { echo 'Missing 4K-data step800 checkpoint'; exit 1; }
for path in "$OUTPUT" "$EVAL" "$FRESH"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite $path"; exit 1; }
done
mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
[[ ${#free_mib[@]} == 4 ]] || exit 1
for free in "${free_mib[@]}"; do
  (( free >= 12000 )) || { echo "Insufficient headroom: $free MiB"; exit 1; }
done
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/train_csra_rollout_output.py \
  --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
  --init-adapter "$INITIAL" --resume "$INITIAL" \
  --data-root eval_results/csra_rollout_data4k_20260912 \
  --output-dir "$OUTPUT" --max-steps 3200 --global-batch-size 256 \
  --learning-rate 1e-4 --variance-weight 0.05 --validate-every 200 \
  --checkpoint-every 800 --seed 20274000
[[ -f "$OUTPUT/complete.json" ]] || exit 1
# If no update improves validation, retain the initializer instead of a worse model.
SELECTED="$INITIAL"
if [[ -f "$OUTPUT/best.pt" ]]; then SELECTED="$OUTPUT/best.pt"; fi
echo "Validation-selected checkpoint: $SELECTED"
ADAPTER_CKPT="$ROOT/$SELECTED" OUT_ROOT="$ROOT/$EVAL" \
  ADAPTER_PRECISION=fp32 BENCHMARK_REPEATS=0 \
  SESSION_PREFIX=rdd_csra_output_data4k_continue3200_best \
  bash 2d/exp/eval_rdd_csra_onpolicy_late6_3seed.sh
# Paired follow-up on the previous fresh seeds, now explicitly not untouched tests.
for seed in 20275001 20275002 20275003 20275004; do
  mkdir -p "$FRESH/seed_$seed"
  CUDA_VISIBLE_DEVICES=3 "$PYTHON" -u 2d/attention_analysis/sample_csra_adapter_preview.py \
    --family rdd --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
    --adapter-ckpt "$SELECTED" \
    --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
    --output-dir "$FRESH/seed_$seed" --seed "$seed" --batch-size 4 \
    --rdd-steps 80,80,90 --benchmark-repeats 0 --adapter-precision fp32 \
    --device cuda:0 2>&1 | tee "$FRESH/seed_$seed/run.log"
done
echo '4K continuation finished and seven quality previews dispatched/completed.'
