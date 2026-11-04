#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2
exec 9>eval_results/csra_prefix8_hidden_audit_20260913.lock
flock -n 9 || { echo 'This exact audit is already active'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
 mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#values[@]} == 4 ]] || return 1
 for value in "${values[@]}"; do ((value>=22000)) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB/card on0–3; unrelated jobs untouched'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
 [[ ! -f "eval_results/csra_prefix8_hidden_audit_20260913/shard_${gpu}.json" ]] || continue
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/audit_csra_prefix_hidden_loss.py --rank "$gpu" \
  > "logs/csra_prefix8_hidden_audit_shard${gpu}_20260913.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/audit_csra_prefix_hidden_loss.py --aggregate
