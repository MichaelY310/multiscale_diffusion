#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_prefix6_quality256_20260913.lock
flock -n 9 || { echo 'This exact prefix6 pilot is already running'; exit 1; }
if tmux has-session -t '=rdd_csra_prefix6_conditional_0123_20260913' 2>/dev/null; then
  echo 'Owned training queue still active; do not overlap'; exit 1
fi
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
"$PY" - <<'PY'
import json,pathlib
for arm in ('aligned','zero'):
 m=json.loads(pathlib.Path(f'checkpoints/RDD-CSRA-prefix6-conditional-{arm}-L400K-20260913/complete.json').read_text())
 assert m['complete'] and not m['smoke'] and m['step']==800 and m['config']['prefix_depth']==6
PY
check_memory() {
 mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#values[@]} == 4 ]] || return 1
 for value in "${values[@]}"; do ((value>=22000)) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB/card on0–3; unrelated jobs untouched'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
 out="eval_results/csra_prefix6_quality256_20260913/shard_${gpu}"
 [[ ! -f "$out/metrics.json" ]] || continue
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_quality_pilot.py \
  --rank "$gpu" --prefix-depth 6 --checkpoint-step 800 --seed-base 20460913 \
  --checkpoint-template 'checkpoints/RDD-CSRA-prefix6-conditional-{arm}-L400K-20260913/best.pt' \
  --output-root eval_results/csra_prefix6_quality256_20260913 \
  > "logs/csra_prefix6_quality256_shard${gpu}_20260913.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/summarize_csra_prefix_quality_pilot.py \
 --root eval_results/csra_prefix6_quality256_20260913 --prefix-depth 6 --checkpoint-step 800 --seed-base 20460913
echo 'Prefix6 quality pilot complete:256new images permethod,768total; not FID or speed'
