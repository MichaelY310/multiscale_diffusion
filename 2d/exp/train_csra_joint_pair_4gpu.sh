#!/usr/bin/env bash
# New experiment only. Never signals/kills another job, never overwrites base.
set -euo pipefail
ROOT=/scr/yuheng/DevOpsBench/multiscale_diffusion
cd "$ROOT"
PYTHON="$ROOT/.venv-rdd-eval/bin/python"
DATA=eval_results/csra_rollout_data4k_20260912
SOURCE=eval_results/csra_joint_sources_20260913
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
check_memory() {
  mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#free_mib[@]} == 4 ]] || return 1
  for free in "${free_mib[@]}"; do (( free >= 25000 )) || return 1; done
}
until check_memory; do echo 'Waiting for >=25,000 MiB free on each of physical GPUs0-3'; sleep 30; done
"$PYTHON" -m unittest discover -s tests -p 'test_csra_*.py'
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u 2d/attention_analysis/collect_csra_joint_sources.py \
    --data-root "$DATA" --output-root "$SOURCE" --shard "$gpu" \
    > "logs/csra_joint_sources_${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
common=(--base-ckpt checkpoints/RDD-Unified-L/0400000.pt
  --init-adapter checkpoints/RDD-CSRA-condition-v2-L400K-20260912/best.pt
  --data-root "$DATA" --source-root "$SOURCE" --global-batch 256
  --adapter-lr 1e-4 --backbone-lr 1e-5 --anchor-weight 1 --variance-weight .05
  --seed 20279000)
for arm in joint frozen; do
  out="checkpoints/RDD-CSRA-jointpilot-${arm}-L400K-20260913"
  [[ ! -f "$out/complete.json" ]] || continue
  until check_memory; do echo 'Waiting for training memory headroom'; sleep 30; done
  # Actual per-card64 smoke, full optimizer allocated; no accumulation.
  smoke="eval_results/csra_joint_smoke_${arm}_20260913"
  if [[ ! -f "$smoke/complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_joint.py "${common[@]}" --arm "$arm" \
      --output-dir "$smoke" --max-steps 2 --validate-every 2 --smoke \
      2>&1 | tee "logs/csra_joint_smoke_${arm}_20260913.log"
  fi
  resume=()
  [[ ! -f "$out/latest.pt" ]] || resume=(--resume "$out/latest.pt")
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_joint.py "${common[@]}" --arm "$arm" \
    --output-dir "$out" --max-steps 800 --validate-every 100 "${resume[@]}" \
    2>&1 | tee -a "logs/csra_joint_${arm}_20260913.log"
done
# A paired preview must use ORIGINAL dense teacher, not dense updated student.
pids=()
for gpu in 0 1 2 3; do
  (
    seeds=(20260921 20260922 20260923 20275001)
    seed_list=("${seeds[$gpu]}")
    [[ "$gpu" != 3 ]] || seed_list=(20275001 20275002 20275003 20275004)
    for seed in "${seed_list[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u 2d/attention_analysis/sample_csra_joint_preview.py \
        --seed "$seed" --output-dir "eval_results/csra_joint_pair_20260913/seed_$seed" \
        2>&1 | tee "logs/csra_joint_preview_${seed}_20260913.log"
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Joint and matched frozen pilot plus all seven paired previews complete.'
