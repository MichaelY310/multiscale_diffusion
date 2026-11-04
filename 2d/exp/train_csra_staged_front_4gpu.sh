#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>eval_results/csra_staged_front_train_20260914.lock
flock -n 9 || { echo 'Exact staged-front job already active';exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
wait_memory() {
  while true; do
    mapfile -t free < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
    ready=1;[[ ${#free[@]} == 4 ]] || ready=0
    for n in "${free[@]}";do ((n >= $1)) || ready=0;done
    ((ready)) && break
    echo "Waiting for$1 MiB free/card on0-3; other jobs untouched";sleep 30
  done
}
for mode in staged direct_control; do
  for arm in aligned zero; do
    out="eval_results/csra_staged_front_${mode}_${arm}_smoke_20260914"
    if [[ ! -f "$out/complete.json" ]]; then
      wait_memory 27000
      CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
        2d/attention_analysis/train_csra_staged_front_pair.py --mode "$mode" --coarse-source "$arm" \
        --output-dir "$out" --smoke 2>&1 | tee "logs/csra_staged_front_${mode}_${arm}_smoke_20260914.log"
    fi
  done
done
"$PY" - <<'PY'
import json
from pathlib import Path
for mode in ('staged','direct_control'):
    folders=[Path(f'eval_results/csra_staged_front_{mode}_{arm}_smoke_20260914') for arm in ('aligned','zero')]
    for folder in folders:
        d=json.loads((folder/'complete.json').read_text())
        assert d['complete'] and d['smoke'] and d['step']==(2 if mode=='staged' else 802)
        if mode=='staged':assert d['warmup_backbone_verified']
    for rank in range(4):
        traces=[json.loads((folder/f'start_trace_rank{rank}.json').read_text()) for folder in folders]
        assert traces[0]==traces[1], 'Unpaired source/zero data sequence'
print('Four real4x64 smokes passed; staged backbone bitwise unchanged; source/zero data matched')
PY
for mode in staged direct_control; do
  for arm in aligned zero; do
    out="checkpoints/RDD-CSRA-front1-${mode}-${arm}-L400K-20260914"
    if [[ -f "$out/complete.json" ]]; then
      "$PY" - "$out/complete.json" "$mode" "$arm" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));assert d['complete'] and not d['smoke'] and d['step']==1600
assert d['args']['mode']==sys.argv[2] and d['args']['coarse_source']==sys.argv[3]
PY
      continue
    fi
    wait_memory 27000
    resume=()
    if [[ -f "$out/latest.pt" ]];then resume=(--resume "$out/latest.pt");
    elif [[ -f "$out/best.pt" ]];then resume=(--resume "$out/best.pt");fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_staged_front_pair.py --mode "$mode" --coarse-source "$arm" \
      --output-dir "$out" "${resume[@]}" 2>&1 | tee -a "logs/csra_staged_front_${mode}_${arm}_20260914.log"
  done
done
"$PY" - <<'PY'
import json
from pathlib import Path
for rank in range(4):
    traces=[]
    for mode in ('staged','direct_control'):
        for arm in ('aligned','zero'):
            folder=Path(f'checkpoints/RDD-CSRA-front1-{mode}-{arm}-L400K-20260914')
            trace=json.loads((folder/f'start_trace_rank{rank}.json').read_text())['trace']
            traces.append([r for r in trace if r['step'] in (800,801)])
    assert all(len(t)==2 for t in traces) and all(t==traces[0] for t in traces), 'Unpaired joint-phase data'
print('All four formal arms match first2joint update indices/t/labels')
PY
for mode in staged direct_control; do
  wait_memory 22000
  root="eval_results/csra_staged_front_${mode}_quality256_20260914"
  template="checkpoints/RDD-CSRA-front1-${mode}-{arm}-L400K-20260914/latest.pt"
  pids=()
  for gpu in 0 1 2 3;do
    [[ ! -f "$root/shard_$gpu/metrics.json" ]] || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_staged_front_quality.py \
      --rank "$gpu" --checkpoint-template "$template" --output-root "$root" \
      > "logs/csra_staged_front_${mode}_quality_gpu${gpu}_20260914.log" 2>&1 &
    pids+=("$!")
  done
  failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
  ((failed==0)) || exit 1
  "$PY" 2d/attention_analysis/summarize_csra_prefix_quality_pilot.py --root "$root" \
    --prefix-depth 6 --checkpoint-step 1600 --seed-base 20460913
done
echo 'Four fixed1600 arms and paired256 pilots complete; review before further training'
