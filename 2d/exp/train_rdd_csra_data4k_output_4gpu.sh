#!/usr/bin/env bash
# A fresh, larger teacher-trajectory corpus; no architecture or loss changes.
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
DATA=eval_results/csra_rollout_data4k_20260912
OUTPUT=checkpoints/RDD-CSRA-output-data4k-L400K-20260912
EVAL=eval_results/csra_output_data4k_best_20260912
INITIAL=checkpoints/RDD-CSRA-output-continue2k-L400K-20260912/best.pt
[[ -f "$INITIAL" ]] || { echo 'Missing validation-selected 800-step initializer'; exit 1; }
for path in "$DATA" "$OUTPUT" "$EVAL"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite $path"; exit 1; }
done
for shard in 0 1 2 3; do
  [[ ! -e "logs/csra_data4k_shard${shard}_20260912.log" ]] || exit 1
done
mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
[[ ${#free_mib[@]} == 4 ]] || exit 1
for free in "${free_mib[@]}"; do
  (( free >= 12000 )) || { echo "Insufficient headroom: ${free} MiB"; exit 1; }
done
mkdir -p "$DATA"
jobs=()
for shard in 0 1 2 3; do
  (
    CUDA_VISIBLE_DEVICES="$shard" OMP_NUM_THREADS=4 "$PYTHON" -u \
      2d/attention_analysis/collect_csra_rollout_dataset.py \
      --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
      --output-dir "$DATA/shard_$shard" --shard "$shard" --shards 4 \
      --batch-size 8 --train-batches 128 --validation-batches 4 \
      --seed 20270000 --check-parity \
      2>&1 | tee "logs/csra_data4k_shard${shard}_20260912.log"
  ) &
  jobs+=("$!")
done
# Wait only for the four children created above. Never kill any process.
failed=0
for job in "${jobs[@]}"; do
  if ! wait "$job"; then failed=1; fi
done
(( failed == 0 )) || { echo 'Collection failed; no training dispatched'; exit 1; }
echo 'All four collection shards finished; trainer checks manifests/parity and disjoint seeds.'
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/train_csra_rollout_output.py \
  --base-ckpt checkpoints/RDD-Unified-L/0400000.pt --init-adapter "$INITIAL" \
  --data-root "$DATA" --output-dir "$OUTPUT" \
  --max-steps 800 --global-batch-size 256 --learning-rate 1e-4 \
  --variance-weight 0.05 --validate-every 100 --checkpoint-every 200 --seed 20274000
[[ -f "$OUTPUT/complete.json" && -f "$OUTPUT/best.pt" ]] || exit 1
ADAPTER_CKPT="$ROOT/$OUTPUT/best.pt" OUT_ROOT="$ROOT/$EVAL" \
  ADAPTER_PRECISION=fp32 SESSION_PREFIX=rdd_csra_output_data4k_best \
  bash 2d/exp/eval_rdd_csra_onpolicy_late6_3seed.sh
echo 'Larger-data training completed; validation-selected three-seed previews dispatched.'
