#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=eval_results/csra_memory_refiner_20260914
PY=.venv-rdd-eval/bin/python
mkdir -p "$ROOT"
exec 9>"$ROOT/train.lock"
flock -n 9 || exit 1
export OMP_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
while [[ ! -f "$ROOT/cost_gate.json" ]]; do
    tmux has-session -t =rdd_csra_memory_refiner_cost_20260914 || { echo 'Cost job exited without gate'; exit 1; }
    sleep 30
done
"$PY" - "$ROOT/cost_gate.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d['complete'] and d['passes_large_video_cost_gate'], 'Cost gate failed: no training authorized by this protocol'
PY
wait_memory() {
    while true; do
        mapfile -t available < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
        ready=1; [[ ${#available[@]} == 4 ]] || ready=0
        for n in "${available[@]}"; do ((n >= $1)) || ready=0; done
        ((ready)) && break
        echo "Waiting for $1 MiB/card; unrelated jobs untouched"; sleep 30
    done
}
wait_memory 27000
if [[ ! -f "$ROOT/train_smoke/complete.json" ]]; then
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
        2d/attention_analysis/train_csra_memory_refiner.py --smoke --output-dir "$ROOT/train_smoke" \
        2>&1 | tee "$ROOT/train_smoke.log"
fi
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('eval_results/csra_memory_refiner_20260914/train_smoke')
old=Path('eval_results/csra_state_replay_20260914/smoke_teacher_only')
a=json.loads((root/'experiment.json').read_text());b=json.loads((old/'experiment.json').read_text())
assert a['initial_validation']==b['initial_validation']
assert a['parent']['zero_init_actual_batch64_max_abs']==0
d=json.loads((root/'complete.json').read_text());assert d['complete'] and d['smoke'] and d['updates']==2
for rank in range(4):
    assert json.loads((root/f'trace_rank{rank}.json').read_text())==json.loads((old/f'trace_rank{rank}.json').read_text())
print('PASS actual4x64 zero-init/initial validation and matched control data order')
PY
OUT=checkpoints/RDD-CSRA-memory-refiner-L400K-20260914
if [[ ! -f "$OUT/complete.json" ]]; then
    wait_memory 27000
    resume=(); [[ ! -f "$OUT/latest.pt" ]] || resume=(--resume "$OUT/latest.pt")
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
        2d/attention_analysis/train_csra_memory_refiner.py --output-dir "$OUT" "${resume[@]}" \
        2>&1 | tee -a "$ROOT/train.log"
fi
wait_memory 20000
pids=()
for gpu in 0 1 2 3; do
    [[ ! -f "$ROOT/quality/shard_$gpu/metrics.json" ]] || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_memory_refiner_quality.py --shard "$gpu" \
        > "$ROOT/quality_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/sample_csra_memory_refiner_quality.py --summarize
echo 'Bounded400 candidate + paired development quality complete; do not automatically extend training.'
