#!/usr/bin/env bash
# A bounded compression-depth test, NOT a new claimed acceleration method.
# Original 8/16 stays frozen; skip 6 rather than 8 fine blocks; 800 updates/arm.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
root=eval_results/csra_prefix6_information_20260913
exec 9>eval_results/csra_prefix6_conditional_pair_20260913.lock
flock -n 9 || { echo 'This exact prefix6 experiment is already running'; exit 1; }
if tmux has-session -t '=rdd_csra_prefix8_hidden_2x2_0123_20260913' 2>/dev/null; then
  echo 'Previous owned training queue is still active; do not overlap'; exit 1
fi
free_kb=$(df -Pk checkpoints | awk 'NR==2 {print $4}')
((free_kb>=45*1024*1024)) || { echo 'Need45GiB disk; nothing deleted'; exit 1; }
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do ((value >= $1)) || return 1; done
}
wait_memory() {
  until check_memory "$1"; do echo "Waiting for ${1}MiB/card on0–3; other jobs untouched"; sleep 30; done
}
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
wait_memory 12000
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/probe_csra_prefix_information.py capture \
    --shard "$gpu" --prefix-depth 6 --output-root "$root" \
    > "logs/csra_prefix6_capture_${gpu}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
wait_memory 12000
CUDA_VISIBLE_DEVICES=0 "$PY" -u 2d/attention_analysis/probe_csra_prefix_information.py fit --output-root "$root" \
  2>&1 | tee logs/csra_prefix6_fit_20260913.log
CUDA_VISIBLE_DEVICES=0 "$PY" -u 2d/attention_analysis/probe_csra_prefix_suffix.py --root "$root" \
  2>&1 | tee logs/csra_prefix6_suffix_20260913.log
"$PY" - "$root/suffix_probe.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));assert m['complete'] and m['oracle_suffix_parity_max_abs']==0 and m['prefix_depth']==6
PY
for mode in aligned zero; do
  wait_memory 26500
  smoke="eval_results/csra_prefix6_conditional_${mode}_smoke_20260913"
  if [[ ! -f "$smoke/complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 6 \
      --bridge-kind conditional_attention --initializer "$root/selected_ridge.pt" \
      --output-dir "$smoke" --max-steps 2 --smoke \
      2>&1 | tee "logs/csra_prefix6_conditional_${mode}_smoke_20260913.log"
  fi
  "$PY" - "$smoke/complete.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));assert m['complete'] and m['smoke'] and m['step']==2 and m['config']['prefix_depth']==6
PY
done
for mode in aligned zero; do
  wait_memory 26500
  out="checkpoints/RDD-CSRA-prefix6-conditional-${mode}-L400K-20260913"
  if [[ -f "$out/complete.json" ]]; then
    "$PY" - "$out/complete.json" "$mode" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));assert m['complete'] and not m['smoke'] and m['step']==800
c=m['config'];assert c['coarse_source']==sys.argv[2] and c['prefix_depth']==6 and c['bridge_kind']=='conditional_attention'
PY
    continue
  fi
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 6 \
    --bridge-kind conditional_attention --initializer "$root/selected_ridge.pt" \
    --output-dir "$out" --max-steps 800 "${resume[@]}" \
    2>&1 | tee -a "logs/csra_prefix6_conditional_${mode}_20260913.log"
done
wait_memory 16000
pids=();seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --prefix-depth 6 --checkpoint-template 'checkpoints/RDD-CSRA-prefix6-conditional-{mode}-L400K-20260913/best.pt' \
    --seed "$seed" --output-dir "eval_results/csra_prefix6_conditional_pair_20260913/seed_$seed" \
    > "logs/csra_prefix6_conditional_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'Prefix6 conditional: matched800 arms and3paired previews complete; inspect before extending.'
