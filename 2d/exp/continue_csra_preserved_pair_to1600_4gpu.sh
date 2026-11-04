#!/usr/bin/env bash
# Bounded continuation of the completed V2 pair; never modifies the 800-step runs.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
check_memory() {
  mapfile -t free_mib < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#free_mib[@]} == 4 ]] || return 1
  for free in "${free_mib[@]}"; do (( free >= 25000 )) || return 1; done
}
common=(--base-ckpt checkpoints/RDD-Unified-L/0400000.pt
  --init-adapter checkpoints/RDD-CSRA-condition-v2-L400K-20260912/best.pt
  --data-root eval_results/csra_rollout_data4k_20260912
  --source-root eval_results/csra_joint_sources_20260913
  --global-batch 256 --adapter-lr 1e-4 --backbone-lr 1e-5
  --anchor-weight 0 --variance-weight .05 --seed 20279000)
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
for arm in preserved_joint preserved_frozen; do
  src="checkpoints/RDD-CSRA-${arm}-L400K-20260913"
  out="checkpoints/RDD-CSRA-${arm}-continue1600-L400K-20260913"
  # Carry BOTH latest optimizer state and historical best. The frozen arm's
  # best is step100, not latest800. Never silently discard that comparison.
  [[ -f "$src/complete.json" && -f "$src/latest.pt" && -f "$src/best.pt" ]]
  mkdir -p "$out"
  for checkpoint in best.pt latest.pt; do
    if [[ ! -f "$out/$checkpoint" ]]; then
      cp --reflink=auto "$src/$checkpoint" "$out/$checkpoint.copying"
      mv -n "$out/$checkpoint.copying" "$out/$checkpoint"
    fi
  done
  if [[ -f "$out/complete.json" ]]; then continue; fi
  until check_memory; do echo 'Waiting for 25,000 MiB/card on physical GPUs0-3'; sleep 30; done
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_joint.py "${common[@]}" --arm "$arm" \
    --output-dir "$out" --resume "$out/latest.pt" --max-steps 1600 --validate-every 100 \
    2>&1 | tee -a "logs/csra_${arm}_continue1600_20260913.log"
done
until check_memory; do echo 'Waiting for paired preview memory headroom'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
  (
    seeds=(20260921 20260922 20260923 20275001)
    seed_list=("${seeds[$gpu]}")
    [[ "$gpu" != 3 ]] || seed_list=(20275001 20275002 20275003 20275004)
    for seed in "${seed_list[@]}"; do
      CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_preserved_preview.py \
        --checkpoint-suffix=-continue1600 --seed "$seed" \
        --output-dir "eval_results/csra_preserved_continue1600_pair_20260913/seed_$seed" \
        2>&1 | tee "logs/csra_preserved_continue1600_preview_${seed}_20260913.log"
    done
  ) &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Bounded V2 continuation to1600 and seven paired previews complete.'
