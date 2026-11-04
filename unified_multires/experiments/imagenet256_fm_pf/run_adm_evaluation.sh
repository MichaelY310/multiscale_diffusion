#!/usr/bin/env bash
# Usage: bash run_adm_evaluation.sh /path/to/completed_sampling_run
set -euo pipefail
if [[ $# -ne 1 ]]; then echo 'Usage: run_adm_evaluation.sh COMPLETED_SAMPLING_RUN_DIR' >&2; exit 2; fi
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd -P)"
SAMPLE_DIR="$(cd -- "$1" && pwd -P)"
ASSET_DIR="$(cd -- "${ADM_ASSET_DIR:-$PROJECT_ROOT/eval_results/reference}" && pwd -P)"
PYTHON_BIN="${ADM_PYTHON:-$PROJECT_ROOT/.venv-adm/bin/python}"
OUT_DIR="$SAMPLE_DIR/evaluation"
mkdir -p "$OUT_DIR"
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS="${ADM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${ADM_THREADS:-4}"
export TF_NUM_INTEROP_THREADS=1
"$PYTHON_BIN" - "$SAMPLE_DIR" "$ASSET_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,assets=map(Path,sys.argv[1:]);summary=json.loads((root/'SUMMARY.json').read_text())
if summary.get('complete') is not True or summary.get('samples')!=50000:
    raise ValueError('Formal evaluation requires a completed50K sampling run, not the64-image preview')
files={p.name for p in (root/'png').glob('*.png')}
if files!={f'{i:06d}.png' for i in range(50000)}:raise ValueError('Missing/extra sample PNGs')
covered=set()
for marker in (root/'batches').glob('*.json'):
    for name,digest in json.loads(marker.read_text())['files'].items():
        covered.add(name)
        if hashlib.sha256((root/'png'/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Sample checksum mismatch: {name}')
if covered!=files:raise ValueError('Batch records do not cover all50K samples')
for name in ('classify_image_graph_def.pb','VIRTUAL_imagenet256_labeled.npz','ADM_ASSETS.json'):
    if not (assets/name).is_file():raise FileNotFoundError(f'Run setup_evaluation.sh first: {name}')
records=json.loads((assets/'ADM_ASSETS.json').read_text())
if {row['file'] for row in records}!={'classify_image_graph_def.pb','VIRTUAL_imagenet256_labeled.npz'}:
    raise ValueError('Incomplete ADM asset manifest')
for row in records:
    path=assets/row['file'];digest=hashlib.sha256()
    if path.stat().st_size!=row['bytes']:raise ValueError(f'Changed ADM asset: {path}')
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):digest.update(block)
    if digest.hexdigest()!=row['sha256']:raise ValueError(f'ADM asset checksum mismatch: {path}')
PY
"$PYTHON_BIN" "$PROJECT_ROOT/2d/png_to_npz.py" "$SAMPLE_DIR/png" \
  --num-samples 50000 --output "$OUT_DIR/samples_50k.part.npz"
mv -- "$OUT_DIR/samples_50k.part.npz" "$OUT_DIR/samples_50k.npz"
# The original evaluator finds the Inception GraphDef in its working directory.
cd "$ASSET_DIR"
"$PYTHON_BIN" "$PROJECT_ROOT/guided-diffusion/evaluations/evaluator.py" \
  "$ASSET_DIR/VIRTUAL_imagenet256_labeled.npz" "$OUT_DIR/samples_50k.npz" \
  2>&1 | tee "$OUT_DIR/adm.log"
"$PYTHON_BIN" - "$OUT_DIR" "$SAMPLE_DIR" "$ASSET_DIR" "$PROJECT_ROOT" <<'PY'
import hashlib,importlib.metadata,json,math,re,sys
from pathlib import Path
out,samples,assets,repo=map(Path,sys.argv[1:]);log=(out/'adm.log').read_text();metrics={}
for name in ('FID','sFID','Inception Score','Precision','Recall'):
    matches=re.findall(r'^'+re.escape(name)+r':\s*([-+\d.eE]+)\s*$',log,re.M)
    if len(matches)!=1 or not math.isfinite(float(matches[0])):raise ValueError(f'Missing/invalid metric: {name}')
    metrics[name]=float(matches[0])
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()
record={'metrics':metrics,'samples':50000,'sampling_run':json.loads((samples/'run.json').read_text()),
        'assets':json.loads((assets/'ADM_ASSETS.json').read_text()),'sample_npz_sha256':sha(out/'samples_50k.npz'),
        'evaluator_sha256':sha(repo/'guided-diffusion/evaluations/evaluator.py'),
        'environment':{n:importlib.metadata.version(n) for n in ('tensorflow-cpu','numpy','scipy')},
        'device':'CPU','sample_npz_removed_after_success':True}
(out/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(metrics,indent=2))
PY
# Reconstructible ~9.83GB intermediate, removed only after all5metrics are saved.
rm -- "$OUT_DIR/samples_50k.npz"
