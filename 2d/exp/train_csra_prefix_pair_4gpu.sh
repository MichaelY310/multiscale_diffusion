#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 25000 )) || return 1; done
}
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
# BOTH real optimizer smokes run before either formal arm starts.
for mode in aligned zero; do
  until check_memory; do echo 'Waiting for25,000MiB/card on0–3'; sleep 30; done
  smoke="eval_results/csra_prefix_v7_${mode}_smoke_20260913"
  if [[ ! -f "$smoke/complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" \
      --output-dir "$smoke" --max-steps 2 --smoke \
      2>&1 | tee "logs/csra_prefix_v7_${mode}_smoke_20260913.log"
  fi
done
for mode in aligned zero; do
  until check_memory; do echo 'Waiting for training headroom'; sleep 30; done
  out="checkpoints/RDD-CSRA-prefix_v7-${mode}-L400K-20260913"
  [[ ! -f "$out/complete.json" ]] || continue
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" \
    --output-dir "$out" --max-steps 800 "${resume[@]}" \
    2>&1 | tee -a "logs/csra_prefix_v7_${mode}_20260913.log"
done
until check_memory; do echo 'Waiting for preview headroom'; sleep 30; done
pids=()
seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --seed "$seed" --output-dir "eval_results/csra_prefix_v7_pair_20260913/seed_$seed" \
    > "logs/csra_prefix_v7_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Prefix V7 pair training and three paired previews complete.'
