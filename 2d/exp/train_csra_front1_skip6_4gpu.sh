#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>eval_results/csra_front1_skip6_train_20260914.lock
flock -n 9 || { echo 'Exact front1 job already active'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
wait_memory() {
  while true; do
    mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
    ready=1
    [[ ${#values[@]} == 4 ]] || ready=0
    for n in "${values[@]}"; do ((n >= $1)) || ready=0; done
    ((ready)) && break
    echo "Waiting for$1 MiB/card on0-3; unrelated jobs untouched"
    sleep 30
  done
}
for mode in aligned zero; do
  smoke="eval_results/csra_front1_skip6_${mode}_smoke_20260914"
  if [[ ! -f "$smoke/complete.json" ]]; then
    wait_memory 27000
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_front_keep_pair.py --coarse-source "$mode" --output-dir "$smoke" \
      --max-steps 2 --smoke 2>&1 | tee "logs/csra_front1_skip6_${mode}_smoke_20260914.log"
  fi
done
"$PY" - <<'PY'
import json
from pathlib import Path
dirs=[Path(f'eval_results/csra_front1_skip6_{m}_smoke_20260914') for m in ('aligned','zero')]
for p in dirs:
    d=json.loads((p/'complete.json').read_text());assert d['complete'] and d['smoke'] and d['step']==2
for rank in range(4):
    traces=[json.loads((p/f'start_trace_rank{rank}.json').read_text()) for p in dirs]
    assert traces[0]==traces[1], 'Unpaired source/zero training order'
print('Both real4x64 smokes and allrank paired data-order checks passed')
PY
for mode in aligned zero; do
  out="checkpoints/RDD-CSRA-front1-skip6-${mode}-L400K-20260914"
  if [[ -f "$out/complete.json" ]]; then
    "$PY" - "$out/complete.json" "$mode" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));assert d['complete'] and not d['smoke'] and d['step']==800 and d['args']['coarse_source']==sys.argv[2]
PY
    continue
  fi
  wait_memory 27000
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
    2d/attention_analysis/train_csra_front_keep_pair.py --coarse-source "$mode" --output-dir "$out" \
    --max-steps 800 "${resume[@]}" 2>&1 | tee -a "logs/csra_front1_skip6_${mode}_20260914.log"
done
wait_memory 22000
root=eval_results/csra_front1_skip6_quality256_20260914
pids=()
for gpu in 0 1 2 3; do
  [[ ! -f "$root/shard_$gpu/metrics.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_front_keep_quality.py --rank "$gpu" \
    > "logs/csra_front1_skip6_quality_gpu${gpu}_20260914.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/summarize_csra_prefix_quality_pilot.py --root "$root" \
  --prefix-depth 6 --checkpoint-step 800 --seed-base 20460913
echo 'Front1 two fixed800 arms and paired256 quality complete; review before more training.'
