#!/usr/bin/env bash
# Bounded convergence check; do not alter the frozen RDD or existing runs.
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
INITIAL=checkpoints/RDD-CSRA-output-pilot-L400K-20260912/0000400.pt
OUTPUT=checkpoints/RDD-CSRA-output-continue2k-L400K-20260912
EVAL=eval_results/csra_output_continue2k_best_20260912

[[ -f "$INITIAL" ]] || { echo 'Missing 400-update checkpoint'; exit 1; }
[[ ! -e "$OUTPUT" && ! -e "$EVAL" ]] || {
  echo 'Refusing to overwrite an existing continuation/evaluation'; exit 1;
}
# Check only the authorized physical GPUs, without signaling any process.
mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
[[ ${#free_mib[@]} == 4 ]] || { echo 'GPU query failed'; exit 1; }
for free in "${free_mib[@]}"; do
  (( free >= 12000 )) || { echo "Insufficient headroom: ${free} MiB"; exit 1; }
done
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
"$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
  2d/attention_analysis/train_csra_rollout_output.py \
  --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
  --init-adapter "$INITIAL" --resume "$INITIAL" \
  --data-root eval_results/csra_rollout_data_20260912 \
  --output-dir "$OUTPUT" --max-steps 2000 \
  --global-batch-size 256 --learning-rate 1e-4 --variance-weight 0.05 \
  --validate-every 100 --checkpoint-every 500 --seed 20262000

[[ -f "$OUTPUT/complete.json" ]] || { echo 'Training did not complete'; exit 1; }
# Select by held-out main-output MSE, never by the displayed sampling seeds.
SELECTED="$INITIAL"
if [[ -f "$OUTPUT/best.pt" ]]; then SELECTED="$OUTPUT/best.pt"; fi
echo "Validation-selected checkpoint: $SELECTED"
ADAPTER_CKPT="$ROOT/$SELECTED" OUT_ROOT="$ROOT/$EVAL" \
  ADAPTER_PRECISION=fp32 SESSION_PREFIX=rdd_csra_output_continue2k_best \
  bash 2d/exp/eval_rdd_csra_onpolicy_late6_3seed.sh
echo 'Continuation finished; three independently logged preview jobs dispatched.'
