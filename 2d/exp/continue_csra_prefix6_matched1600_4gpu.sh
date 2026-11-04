#!/usr/bin/env bash
# Only the training budget changes: prefix6 best800 -> max1600 in both arms.
# Keeps every old checkpoint and result. No unbounded automatic continuation.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>eval_results/csra_prefix6_conditional1600_20260913.lock
flock -n 9 || { echo 'This exact continuation is already running'; exit 1; }
while tmux has-session -t '=rdd_csra_prefix_video_feasibility_20260913' 2>/dev/null; do
    echo 'Waiting for our own video timing job to finish; do not contaminate its timings.'
    sleep 30
done
"$PY" - <<'PY'
import json
m=json.load(open('eval_results/csra_prefix_video_feasibility_20260913/full/metrics.json'))
assert m['complete'] and not m['smoke'] and len(m['entries'])==8
assert m['baseline_wrapper_max_abs']==0 and m['duplicate_model_max_abs']==0
eligible=[e for e in m['entries'] if e['prefix_depth']==6 and e['case'].startswith('1280x768')]
assert len(eligible)==2 and all(e['latency_reduction_percent']['median']>=10 for e in eligible), 'Review video overhead before more training'
for mode in ('aligned','zero'):
    p=f'checkpoints/RDD-CSRA-prefix6-conditional-{mode}-L400K-20260913/complete.json'
    c=json.load(open(p));assert c['complete'] and not c['smoke'] and c['step']==800
    assert c['config']['prefix_depth']==6 and c['config']['coarse_source']==mode
print('Matched prefix6 parents and bounded video feasibility gates passed; quality remains unproven.')
PY
free_kb=$(df -Pk checkpoints | awk 'NR==2 {print $4}')
((free_kb>=45*1024*1024)) || { echo 'Need45GiB disk; nothing deleted'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
wait_memory() {
  while true; do
    mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
    ready=1
    [[ ${#values[@]} == 4 ]] || ready=0
    for value in "${values[@]}"; do ((value >= $1)) || ready=0; done
    ((ready==1)) && return
    echo "Waiting for ${1}MiB/card on0–3; all other jobs untouched"
    sleep 30
  done
}
for mode in aligned zero; do
    out="checkpoints/RDD-CSRA-prefix6-conditional1600-${mode}-L400K-20260913"
    parent="checkpoints/RDD-CSRA-prefix6-conditional-${mode}-L400K-20260913/best.pt"
    if [[ -f "$out/complete.json" ]]; then
        "$PY" - "$out/complete.json" "$mode" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));assert m['complete'] and not m['smoke'] and m['step']==1600
assert m['config']['coarse_source']==sys.argv[2] and m['config']['prefix_depth']==6
PY
        continue
    fi
    wait_memory 26500
    mkdir -p "$out"
    # Retain the parent as a legitimate best candidate if continuation worsens.
    # cp writes a NEW owned filename only; old checkpoints are never changed.
    if [[ ! -f "$out/best.pt" ]]; then cp --reflink=auto "$parent" "$out/best.pt"; fi
    resume="$parent"
    if [[ -f "$out/latest.pt" ]]; then resume="$out/latest.pt"; fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_prefix_pair.py --coarse-source "$mode" --prefix-depth 6 \
      --bridge-kind conditional_attention --initializer eval_results/csra_prefix6_information_20260913/selected_ridge.pt \
      --output-dir "$out" --max-steps 1600 --resume "$resume" \
      2>&1 | tee -a "logs/csra_prefix6_conditional1600_${mode}_20260913.log"
done
wait_memory 16000
pids=(); seeds=(20260921 20260923 20275004)
for gpu in 0 1 2; do
  seed="${seeds[$gpu]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --prefix-depth 6 --checkpoint-template 'checkpoints/RDD-CSRA-prefix6-conditional1600-{mode}-L400K-20260913/best.pt' \
    --seed "$seed" --output-dir "eval_results/csra_prefix6_conditional1600_pair_20260913/seed_$seed" \
    > "logs/csra_prefix6_conditional1600_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'Bounded matched1600 continuation plus3development previews complete; review before any more training.'
