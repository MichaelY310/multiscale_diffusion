#!/usr/bin/env bash
# New mild target: preserve all old prefix14 artifacts and checkpoints.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
root=eval_results/csra_prefix8_information_20260913
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= $1 )) || return 1; done
}
wait_memory() {
  until check_memory "$1"; do echo "Waiting for ${1}MiB/card on physical0–3; other jobs untouched"; sleep 30; done
}
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
wait_memory 12000
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_prefix_information.py capture \
    --shard "$gpu" --prefix-depth 8 --output-root "$root" \
    > "logs/csra_prefix8_capture_${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
CUDA_VISIBLE_DEVICES=0 "$PY" -u 2d/attention_analysis/probe_csra_prefix_information.py fit --output-root "$root" \
  2>&1 | tee logs/csra_prefix8_fit_20260913.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u 2d/attention_analysis/probe_csra_prefix_suffix.py --root "$root" \
  2>&1 | tee logs/csra_prefix8_suffix_20260913.log
# Oracle exact suffix parity is required before either training arm starts.
"$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["complete"] and d["oracle_suffix_parity_max_abs"] == 0 and d["prefix_depth"] == 8' "$root/suffix_probe.json"
for mode in aligned zero; do
  wait_memory 25000
  smoke="eval_results/csra_prefix8_${mode}_smoke_20260913"
  if [[ ! -f "$smoke/complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 8 \
      --initializer "$root/selected_ridge.pt" --output-dir "$smoke" --max-steps 2 --smoke \
      2>&1 | tee "logs/csra_prefix8_${mode}_smoke_20260913.log"
  fi
done
for mode in aligned zero; do
  wait_memory 25000
  out="checkpoints/RDD-CSRA-prefix8-${mode}-L400K-20260913"
  [[ ! -f "$out/complete.json" ]] || continue
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 8 \
    --initializer "$root/selected_ridge.pt" --output-dir "$out" --max-steps 1600 "${resume[@]}" \
    2>&1 | tee -a "logs/csra_prefix8_${mode}_20260913.log"
done
wait_memory 25000
pids=()
seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --prefix-depth 8 --checkpoint-template 'checkpoints/RDD-CSRA-prefix8-{mode}-L400K-20260913/best.pt' \
    --seed "$seed" --output-dir "eval_results/csra_prefix8_pair_20260913/seed_$seed" \
    > "logs/csra_prefix8_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Prefix8 new targets, ridge, suffix parity, matched1600 training and3paired previews complete.'
