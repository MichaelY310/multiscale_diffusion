#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
ROOT=eval_results/csra_state_replay_20260914
mkdir -p "$ROOT"
exec 9>"$ROOT/train.lock"
flock -n 9 || { echo 'Exact replay training already running'; exit 1; }
export OMP_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
while true; do
    complete=1
    for gpu in 0 1 2 3; do [[ -f "$ROOT/data/shard_$gpu/manifest.json" ]] || complete=0; done
    ((complete)) && break
    if ! tmux has-session -t =rdd_csra_state_replay_0123_20260914 2>/dev/null; then
        echo 'Collector exited before all data complete; inspect its exact log'; exit 1
    fi
    echo 'Waiting for complete paired replay data'; sleep 30
done
"$PY" 2d/attention_analysis/summarize_csra_state_replay.py --data-only
wait_memory() {
    while true; do
        mapfile -t available < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
        ready=1; [[ ${#available[@]} == 4 ]] || ready=0
        for n in "${available[@]}"; do ((n >= $1)) || ready=0; done
        ((ready)) && break
        echo "Waiting for$1MiB/card; unrelated tasks untouched"; sleep 30
    done
}
for arm in teacher_only mixed50; do
    out="$ROOT/smoke_$arm"
    [[ ! -f "$out/complete.json" ]] || continue
    wait_memory 27000
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
        2d/attention_analysis/train_csra_state_replay.py --arm "$arm" --output-dir "$out" --smoke \
        2>&1 | tee "$ROOT/smoke_$arm.log"
done
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('eval_results/csra_state_replay_20260914')
arms=['teacher_only','mixed50']
configs=[json.loads((root/f'smoke_{a}/experiment.json').read_text()) for a in arms]
assert configs[0]['initial_validation']==configs[1]['initial_validation']
for a in arms:
    d=json.loads((root/f'smoke_{a}/complete.json').read_text())
    assert d['complete'] and d['smoke'] and d['updates']==2
for rank in range(4):
    ts=[json.loads((root/f'smoke_{a}/trace_rank{rank}.json').read_text())['trace'] for a in arms]
    assert len(ts[0])==len(ts[1])==2
    for i in range(2):
        for k in ('update','indices','labels','t'):assert ts[0][i][k]==ts[1][i][k]
        assert not any(ts[0][i]['state_role'])
        assert ts[1][i]['state_role']==[j%2 for j in ts[1][i]['indices']]
print('Both actual4x64 smokes passed; identical parent/initial validation/indices/labels/t. Only state domain differs.')
PY
for arm in teacher_only mixed50; do
    out="checkpoints/RDD-CSRA-state-replay-${arm}-L400K-20260914"
    if [[ -f "$out/complete.json" ]]; then
        "$PY" - "$out/complete.json" "$arm" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));assert d['complete'] and not d['smoke'] and d['updates']==400 and d['arm']==sys.argv[2]
PY
        continue
    fi
    wait_memory 27000
    resume=();[[ ! -f "$out/latest.pt" ]] || resume=(--resume "$out/latest.pt")
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
        2d/attention_analysis/train_csra_state_replay.py --arm "$arm" --output-dir "$out" "${resume[@]}" \
        2>&1 | tee -a "$ROOT/train_$arm.log"
done
wait_memory 20000
pids=()
for gpu in 0 1 2 3; do
    [[ ! -f "$ROOT/quality/shard_$gpu/metrics.json" ]] || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_state_replay_quality.py --shard "$gpu" \
        > "$ROOT/quality_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
failed=0;for pid in "${pids[@]}";do wait "$pid" || failed=1;done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/summarize_csra_state_replay.py
echo 'Fixed400x2 matched replay pilot and1024PNG complete; inspect quality, no automatic further training.'
