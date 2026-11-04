#!/usr/bin/env bash
# Independent fixed-checkpoint distribution-quality evaluation; never training.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
ROOT=eval_results/csra_reader_independent_fid10k_20260914
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$ROOT"
exec 9>"$ROOT/run.lock"
flock -n 9 || { echo 'This exact evaluation is already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_reader_fid10k.py'
"$PY" - <<'PY'
import json,pathlib,shutil,sys
sys.path.insert(0,'2d/attention_analysis')
from sample_csra_reader_fid10k import ROOT,READER,READER_SHA,BASE_SHA,sha,write_json
assert shutil.disk_usage('.').free > 24*2**30, 'Need24GiB disk; do not delete other results'
mem=dict(line.split(':',1) for line in pathlib.Path('/proc/meminfo').read_text().splitlines())
assert int(mem['MemAvailable'].split()[0])*1024 > 16*2**30, 'Need16GiB available CPU RAM'
assert sha(READER)==READER_SHA
assert sha('checkpoints/RDD-Unified-L/0400000.pt')==BASE_SHA
paths=['classify_image_graph_def.pb','eval_results/reference/VIRTUAL_imagenet256_labeled.npz',
       'guided-diffusion/evaluations/evaluator.py','2d/diagnostics/evaluate_many_npz.py']
record={p:sha(p) for p in paths}
dest=ROOT/'metric_provenance.json'
if dest.exists(): assert json.loads(dest.read_text())==record, 'Metric/reference changed'
write_json(dest,record)
PY
check_memory() {
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=22528)) || return 1; done
}
run_workers() {
 local phase="$1" target="$2"
 local -a extra=() pids=()
 [[ "$phase" != smoke ]] || extra=(--smoke)
 until check_memory; do echo 'Waiting for22GiB free on0-3; other tasks untouched'; sleep 30; done
 mkdir -p "$target"
 for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_reader_fid10k.py sample \
   --root "$target" --rank "$gpu" "${extra[@]}" >> "$target/sample_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
 done
 local failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 ((failed==0)) || { echo "$phase worker failed; inspect exact logs, no unrelated signals"; return 1; }
 "$PY" 2d/attention_analysis/sample_csra_reader_fid10k.py assemble --root "$target" "${extra[@]}"
 echo "$phase: all paired chunks, coarse endpoints, labels, NPZ shape and SHA verified"
}
run_workers smoke "$ROOT/smoke"
run_workers formal "$ROOT"
# Same ADM metric implementation for both10K arms, reference reused on CPU.
CUDA_VISIBLE_DEVICES='' TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 \
 "$PY" -u 2d/diagnostics/evaluate_many_npz.py --repo-root "$PWD" \
 --reference "$PWD/eval_results/reference/VIRTUAL_imagenet256_labeled.npz" \
 --sample "original_rdd=$PWD/$ROOT/original_rdd.npz" \
 --sample "reader=$PWD/$ROOT/reader.npz" \
 --output-dir "$ROOT/metrics" --name-suffix fid10k \
 2>&1 | tee -a "$ROOT/metrics.log"
"$PY" - <<'PY'
import json,math,pathlib,sys
sys.path.insert(0,'2d/attention_analysis')
from sample_csra_reader_fid10k import ROOT,ARMS,sha,write_json
provenance=json.loads((ROOT/'metric_provenance.json').read_text())
assert all(sha(p)==s for p,s in provenance.items()), 'Changed reference/evaluator'
summary=ROOT/'metrics/summary_fid10k.json'
data=json.loads(summary.read_text())['datasets']
assert set(data)==set(ARMS)
assert all(set(v)=={'Inception Score','FID','sFID','Precision','Recall'} and
           all(math.isfinite(x) for x in v.values()) for v in data.values())
write_json(ROOT/'complete.json',dict(complete=True,samples_per_arm=10000,
    summary_sha256=sha(summary),metric_provenance=provenance,
    warning='Independent same-N10K paired estimate; not final50K or a speed benchmark.'))
print('Completed10K per arm with IS/FID/sFID/Precision/Recall. No further sampling scheduled.')
PY
