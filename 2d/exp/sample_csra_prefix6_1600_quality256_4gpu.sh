#!/usr/bin/env bash
# Repeat the SAME fixed 256 development labels/noises as prefix6 best800.
# This is a paired budget check, NOT a fresh held-out test or more training.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_prefix6_1600_quality256_20260914.lock
flock -n 9 || { echo 'This exact paired budget evaluation is already running'; exit 1; }
if tmux has-session -t '=rdd_csra_prefix6_matched1600_0123_20260913' 2>/dev/null; then
  echo 'Owned continuation still active; do not overlap'; exit 1
fi
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
"$PY" - <<'PY'
import json,pathlib
for arm in ('aligned','zero'):
 p=pathlib.Path(f'checkpoints/RDD-CSRA-prefix6-conditional1600-{arm}-L400K-20260913/complete.json')
 m=json.loads(p.read_text());assert m['complete'] and not m['smoke'] and m['step']==1600
 assert m['config']['prefix_depth']==6 and m['config']['coarse_source']==arm
old=json.load(open('eval_results/csra_prefix6_quality256_20260913/summary.json'))
assert old['complete'] and old['prefix_depth']==6 and old['checkpoint_step']==800 and old['seed_base']==20460913
PY
check_memory() {
 mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#values[@]} == 4 ]] || return 1
 for value in "${values[@]}"; do ((value>=22000)) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB/card on0–3; unrelated jobs untouched'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
 out="eval_results/csra_prefix6_1600_quality256_20260914/shard_${gpu}"
 [[ ! -f "$out/metrics.json" ]] || continue
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_quality_pilot.py \
  --rank "$gpu" --prefix-depth 6 --checkpoint-step 1600 --seed-base 20460913 \
  --checkpoint-template 'checkpoints/RDD-CSRA-prefix6-conditional1600-{arm}-L400K-20260913/best.pt' \
  --output-root eval_results/csra_prefix6_1600_quality256_20260914 \
  > "logs/csra_prefix6_1600_quality256_shard${gpu}_20260914.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/summarize_csra_prefix_quality_pilot.py \
 --root eval_results/csra_prefix6_1600_quality256_20260914 --prefix-depth 6 --checkpoint-step 1600 --seed-base 20460913
echo 'Paired best800->1600 development evaluation complete; verify identical dense hashes before comparing budgets.'
