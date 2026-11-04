#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
ROOT=eval_results/csra_memory_refiner_20260914
mkdir -p "$ROOT"
exec 9>"$ROOT/cost.lock"
flock -n 9 || { echo 'This exact benchmark is already running'; exit 1; }
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES= .venv-rdd-eval/bin/python -m unittest discover -s tests -p test_csra_memory_refiner.py -v
while true; do
    available=$(nvidia-smi -i 0 --query-gpu=memory.free --format=csv,noheader,nounits)
    [[ "$available" =~ ^[[:space:]]*[0-9]+[[:space:]]*$ ]] || exit 1
    ((available >= 33000)) && break
    echo 'Waiting for 33,000 MiB on GPU0; never touching other jobs'; sleep 30
done
CUDA_VISIBLE_DEVICES=0 .venv-rdd-video/bin/python -u rdd_video/benchmark_csra_prefix_feasibility.py \
    --output "$ROOT/video_cost" --port 29743 --front-keep 1 --memory-refine-depth 2 \
    --memory-fraction .24 --min-free-gib 32 --groups 8 --warmups 2 \
    2>&1 | tee "$ROOT/video_cost.log"
.venv-rdd-eval/bin/python - "$ROOT" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]); d=json.loads((p/'video_cost/metrics.json').read_text())
assert d['complete'] and not d['smoke'] and len(d['entries'])==8
assert d['duplicate_model_max_abs']==d['baseline_wrapper_max_abs']==d['zero_init_refiner_final_latent_max_abs']==0
selected=[r for r in d['entries'] if r['variant']=='reader_refined_memory' and r['case'].startswith('1280')]
assert len(selected)==2
# A conservative screening gate, not a claim of trained-video quality or speed.
ok=all(r['paired_latency_ci95_percent']['lower']>=10 for r in selected)
result=dict(complete=True,passes_large_video_cost_gate=ok,entries=[dict(case=r['case'],ci=r['paired_latency_ci95_percent']) for r in selected],
    warning='Untrained branch, shared GPU. Lower CI>=10% on both1280 cases only allows a training pilot; not final acceptance.')
(p/'cost_gate.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
PY
