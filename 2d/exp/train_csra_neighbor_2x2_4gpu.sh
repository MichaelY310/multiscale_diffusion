#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=2 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
ROOT=eval_results/csra_neighbor_2x2_20260914
mkdir -p "$ROOT"
exec 9>"$ROOT/run.lock"
flock -n 9 || { echo 'Exact neighbor pilot already active'; exit 1; }
"$PY" rdd_video/audit_csra_neighbor_video.py > "$ROOT/video_timing_gate.json"
CUDA_VISIBLE_DEVICES=0 "$PY" -m unittest discover -s tests -p 'test_csra_neighbor*.py' -v 2>&1 | tee "$ROOT/unit_tests.log"
wait_memory() {
    while true; do
        mapfile -t free < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
        ready=1; [[ ${#free[@]} == 4 ]] || ready=0
        for n in "${free[@]}"; do ((n >= $1)) || ready=0; done
        ((ready)) && break
        echo "Waiting for$1MiB/card on0-3; other jobs untouched"; sleep 30
    done
}
for neighbor in off on; do
    for source in aligned zero; do
        out="$ROOT/smoke_${neighbor}_${source}"
        if [[ ! -f "$out/complete.json" ]]; then
            wait_memory 27000
            CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
                2d/attention_analysis/train_csra_neighbor_pair.py --neighbor "$neighbor" --coarse-source "$source" \
                --output-dir "$out" --smoke 2>&1 | tee "$ROOT/smoke_${neighbor}_${source}.log"
        fi
    done
done
"$PY" - <<'PY'
import json
from pathlib import Path
root=Path('eval_results/csra_neighbor_2x2_20260914')
folders=[root/f'smoke_{n}_{s}' for n in ('off','on') for s in ('aligned','zero')]
for folder in folders:
    d=json.loads((folder/'complete.json').read_text())
    assert d['complete'] and d['smoke'] and d['step']==1602 and d['initial_branch_max_abs']==0
for rank in range(4):
    traces=[json.loads((p/f'start_trace_rank{rank}.json').read_text()) for p in folders]
    assert len(traces[0]['trace'])==2 and all(t==traces[0] for t in traces)
for source in ('aligned','zero'):
    a,b=[json.loads((root/f'smoke_{n}_{source}'/'experiment.json').read_text()) for n in ('off','on')]
    assert a['parent']==b['parent'] and a['initial_validation']==b['initial_validation']
print('Four actual4x64 smokes passed: parent function exact, all indices/t/labels paired, initial validation identical on/off')
PY
for neighbor in off on; do
    for source in aligned zero; do
        out="checkpoints/RDD-CSRA-neighbor-${neighbor}-${source}-L400K-20260914"
        if [[ -f "$out/complete.json" ]]; then
            "$PY" - "$out/complete.json" "$neighbor" "$source" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));assert d['complete'] and not d['smoke'] and d['step']==2400
assert d['args']['neighbor']==sys.argv[2] and d['args']['coarse_source']==sys.argv[3]
PY
            continue
        fi
        wait_memory 27000
        resume=(); if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt"); fi
        CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
            2d/attention_analysis/train_csra_neighbor_pair.py --neighbor "$neighbor" --coarse-source "$source" \
            --output-dir "$out" "${resume[@]}" 2>&1 | tee -a "$ROOT/train_${neighbor}_${source}.log"
    done
done
"$PY" - <<'PY'
import json
from pathlib import Path
folders=[Path(f'checkpoints/RDD-CSRA-neighbor-{n}-{s}-L400K-20260914') for n in ('off','on') for s in ('aligned','zero')]
for rank in range(4):
    rows=[json.loads((f/f'start_trace_rank{rank}.json').read_text()) for f in folders]
    assert len(rows[0]['trace'])==2 and all(r==rows[0] for r in rows)
print('All four formal starts paired')
PY
for neighbor in off on; do
    wait_memory 22000
    root="eval_results/csra_neighbor_${neighbor}_quality256_20260914"
    template="checkpoints/RDD-CSRA-neighbor-${neighbor}-{arm}-L400K-20260914/latest.pt"
    pids=()
    for gpu in 0 1 2 3; do
        [[ ! -f "$root/shard_$gpu/metrics.json" ]] || continue
        CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_neighbor_quality.py \
            --rank "$gpu" --checkpoint-template "$template" --output-root "$root" \
            > "$ROOT/quality_${neighbor}_gpu${gpu}.log" 2>&1 &
        pids+=("$!")
    done
    failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
    ((failed==0)) || exit 1
    "$PY" 2d/attention_analysis/summarize_csra_prefix_quality_pilot.py --root "$root" \
        --prefix-depth 6 --checkpoint-step 2400 --seed-base 20460913
done
"$PY" 2d/attention_analysis/summarize_csra_neighbor_quality.py
echo 'Four fixed2400 pilots and paired quality complete; inspect interaction and all grids before extending'
