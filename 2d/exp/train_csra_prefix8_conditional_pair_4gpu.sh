#!/usr/bin/env bash
# New architecture, same starting affine function and paired train budget.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
root=eval_results/csra_prefix8_information_20260913
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 25000 )) || return 1; done
}
wait_memory() { until check_memory; do echo 'Waiting for25GiB/card on0–3; other jobs untouched'; sleep 30; done; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
for mode in aligned zero; do
  wait_memory
  smoke="eval_results/csra_prefix8_conditional_${mode}_smoke_20260913"
  [[ ! -f "$smoke/complete.json" ]] || continue
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 8 \
    --bridge-kind conditional_attention --initializer "$root/selected_ridge.pt" \
    --output-dir "$smoke" --max-steps 2 --smoke \
    2>&1 | tee "logs/csra_prefix8_conditional_${mode}_smoke_20260913.log"
done
for mode in aligned zero; do
  wait_memory
  out="checkpoints/RDD-CSRA-prefix8-conditional-${mode}-L400K-20260913"
  [[ ! -f "$out/complete.json" ]] || continue
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 8 \
    --bridge-kind conditional_attention --initializer "$root/selected_ridge.pt" \
    --output-dir "$out" --max-steps 1600 "${resume[@]}" \
    2>&1 | tee -a "logs/csra_prefix8_conditional_${mode}_20260913.log"
done
wait_memory
pids=()
seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  out="eval_results/csra_prefix8_conditional_pair_20260913/seed_${seed}"
  [[ ! -f "$out/metrics.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --prefix-depth 8 --checkpoint-template 'checkpoints/RDD-CSRA-prefix8-conditional-{mode}-L400K-20260913/best.pt' \
    --seed "$seed" --output-dir "$out" > "logs/csra_prefix8_conditional_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Conditional-prefix matched training and3paired previews complete'
