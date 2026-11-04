#!/usr/bin/env bash
# Coarse full-hidden h readout, both arms start from the SAME zero-source V2.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
check_memory() {
  mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#free_mib[@]} == 4 ]] || return 1
  for free in "${free_mib[@]}"; do (( free >= 25000 )) || return 1; done
}
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
until check_memory; do echo 'Waiting for 25,000MiB/card, only physical GPUs0-3'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/collect_csra_hidden_sources.py \
    --data-root eval_results/csra_rollout_data4k_20260912 \
    --source-root eval_results/csra_joint_sources_20260913 \
    --output-root eval_results/csra_hidden_sources_20260913 --shard "$gpu" \
    > "logs/csra_hidden_source_${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
common=(--base-ckpt checkpoints/RDD-Unified-L/0400000.pt
  --init-adapter checkpoints/RDD-CSRA-condition-v2-L400K-20260912/best.pt
  --init-student checkpoints/RDD-CSRA-preserved_joint-no_coarse-L400K-20260913/best.pt
  --data-root eval_results/csra_rollout_data4k_20260912
  --source-root eval_results/csra_joint_sources_20260913
  --hidden-root eval_results/csra_hidden_sources_20260913 --arm preserved_joint
  --adapter-architecture hidden_memory_v6 --global-batch 256
  --adapter-lr 1e-4 --backbone-lr 1e-5 --anchor-weight 0 --variance-weight .05 --seed 20279000)
for mode in aligned zero; do
  until check_memory; do echo 'Waiting for training headroom'; sleep 30; done
  smoke="eval_results/csra_hidden_v6_${mode}_smoke_20260913"
  if [[ ! -f "$smoke/complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_joint.py "${common[@]}" --coarse-source "$mode" \
      --output-dir "$smoke" --max-steps 2 --validate-every 2 --smoke \
      2>&1 | tee "logs/csra_hidden_v6_${mode}_smoke_20260913.log"
  fi
  out="checkpoints/RDD-CSRA-hidden_v6-${mode}-L400K-20260913"
  [[ ! -f "$out/complete.json" ]] || continue
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  until check_memory; do echo 'Waiting for formal training headroom'; sleep 30; done
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_joint.py "${common[@]}" --coarse-source "$mode" \
    --output-dir "$out" --max-steps 800 --validate-every 100 "${resume[@]}" \
    2>&1 | tee -a "logs/csra_hidden_v6_${mode}_20260913.log"
done
until check_memory; do echo 'Waiting for preview headroom'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
  (
    seeds=(20260921 20260922 20260923 20275001)
    seed_list=("${seeds[$gpu]}")
    [[ "$gpu" != 3 ]] || seed_list=(20275001 20275002 20275003 20275004)
    for seed in "${seed_list[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_preserved_preview.py \
        --hidden-control --seed "$seed" --output-dir "eval_results/csra_hidden_pair_20260913/seed_$seed" \
        2>&1 | tee "logs/csra_hidden_preview_${seed}_20260913.log"
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Full-hidden coarse memory and matched trained zero control complete.'
