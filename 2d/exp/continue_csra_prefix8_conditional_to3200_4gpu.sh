#!/usr/bin/env bash
# Bounded budget test, NOT a new architecture. Preserve both1600-step parents.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>eval_results/csra_prefix8_conditional3200_20260913.lock
flock -n 9 || { echo 'This exact continuation is already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
free_kb=$(df -Pk checkpoints | awk 'NR==2 {print $4}')
(( free_kb >= 30*1024*1024 )) || { echo 'Need30GiB disk headroom; nothing deleted'; exit 1; }
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 25000 )) || return 1; done
}
wait_memory() { until check_memory; do echo 'Waiting for25GiB/card on0–3; other jobs untouched'; sleep 30; done; }
for mode in aligned zero; do
  parent="checkpoints/RDD-CSRA-prefix8-conditional-${mode}-L400K-20260913"
  out="checkpoints/RDD-CSRA-prefix8-conditional3200-${mode}-L400K-20260913"
  "$PY" - "$parent/complete.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['complete'] and d['step']==1600 and not d['smoke'], d
assert d['config']['bridge_kind']=='conditional_attention' and d['config']['prefix_depth']==8
PY
  if [[ -f "$out/complete.json" ]]; then
    "$PY" - "$out/complete.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); assert d['complete'] and d['step']==3200 and not d['smoke'], d
PY
    continue
  fi
  wait_memory
  mkdir -p "$out"
  # Seed best so even a non-improving continuation has an auditable fallback.
  # Original parent files are never changed. Complete CPU load + existing resume
  # guards verify source, initializer, depth, data, seed, optimizer and format.
  if [[ ! -f "$out/best.pt" && ! -f "$out/latest.pt" ]]; then
    cp --reflink=auto "$parent/best.pt" "$out/best.pt"
    cmp -s "$parent/best.pt" "$out/best.pt"
  fi
  resume="$out/best.pt"
  [[ ! -f "$out/latest.pt" ]] || resume="$out/latest.pt"
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 8 \
    --bridge-kind conditional_attention \
    --initializer eval_results/csra_prefix8_information_20260913/selected_ridge.pt \
    --output-dir "$out" --resume "$resume" --max-steps 3200 \
    2>&1 | tee -a "logs/csra_prefix8_conditional3200_${mode}_20260913.log"
done
wait_memory
pids=();seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  out="eval_results/csra_prefix8_conditional3200_pair_20260913/seed_${seed}"
  [[ ! -f "$out/metrics.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --prefix-depth 8 \
    --checkpoint-template 'checkpoints/RDD-CSRA-prefix8-conditional3200-{mode}-L400K-20260913/best.pt' \
    --seed "$seed" --output-dir "$out" > "logs/csra_prefix8_conditional3200_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Conditional-prefix3200 matched continuation and3paired previews complete'
