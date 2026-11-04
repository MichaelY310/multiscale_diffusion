#!/usr/bin/env bash
# Matched V2-vs-current-condition V3, then all paired development images.
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
INITIAL=checkpoints/RDD-CSRA-output-data4k-continue3200-L400K-20260912/best.pt
[[ -f "$INITIAL" ]] || exit 1
for arm in v2 v3; do
  for path in "checkpoints/RDD-CSRA-condition-${arm}-L400K-20260912" "eval_results/csra_condition_${arm}_20260912"; do
    [[ ! -e "$path" ]] || { echo "Refusing to overwrite $path"; exit 1; }
  done
done
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
for arm in v2 v3; do
  mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#free_mib[@]} == 4 ]] || exit 1
  for free in "${free_mib[@]}"; do (( free >= 12000 )) || exit 1; done
  arch=fine_context_v2
  [[ "$arm" != v3 ]] || arch=teacher_condition_v3
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_rollout_output.py \
    --base-ckpt checkpoints/RDD-Unified-L/0400000.pt --init-adapter "$INITIAL" \
    --adapter-architecture "$arch" --data-root eval_results/csra_rollout_data4k_20260912 \
    --output-dir "checkpoints/RDD-CSRA-condition-${arm}-L400K-20260912" \
    --max-steps 800 --global-batch-size 256 --learning-rate 1e-4 --variance-weight 0.05 \
    --validate-every 100 --checkpoint-every 400 --seed 20276000 \
    2>&1 | tee "logs/csra_condition_${arm}_20260912.log"
  [[ -f "checkpoints/RDD-CSRA-condition-${arm}-L400K-20260912/complete.json" ]] || exit 1
done
# Finish both distributed jobs first; previews then occupy exactly GPUs0-3.
pids=()
for gpu in 0 1 2 3; do
  (
    seeds=(20260921 20260922 20260923 20275001)
    for arm in v2 v3; do
      seed_list=("${seeds[$gpu]}")
      [[ "$gpu" != 3 ]] || seed_list=(20275001 20275002 20275003 20275004)
      for seed in "${seed_list[@]}"; do
        out="eval_results/csra_condition_${arm}_20260912/seed_$seed"
        mkdir -p "$out"
        CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u 2d/attention_analysis/sample_csra_adapter_preview.py \
          --family rdd --base-ckpt checkpoints/RDD-Unified-L/0400000.pt \
          --adapter-ckpt "checkpoints/RDD-CSRA-condition-${arm}-L400K-20260912/best.pt" \
          --vae-path checkpoints/pretrained/stabilityai/sd-vae-ft-ema \
          --output-dir "$out" --seed "$seed" --batch-size 4 --rdd-steps 80,80,90 \
          --benchmark-repeats 0 --adapter-precision fp32 --device cuda:0 \
          2>&1 | tee "$out/run.log"
      done
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Matched V2/V3 training and all fourteen paired development previews complete.'
