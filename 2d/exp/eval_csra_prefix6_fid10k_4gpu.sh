#!/usr/bin/env bash
# Three matched fixed-checkpoint arms; no training, no alteration to old runs.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
ROOT=eval_results/csra_prefix6_fid10k_20260914
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec 9>eval_results/csra_prefix6_fid10k_20260914.lock
flock -n 9 || { echo 'This exact evaluation is already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
"$PY" - <<'PY'
import shutil,pathlib,json
assert shutil.disk_usage('.').free > 25*2**30, 'Need25GiB disk'
assert pathlib.Path('classify_image_graph_def.pb').is_file()
assert pathlib.Path('eval_results/reference/VIRTUAL_imagenet256_labeled.npz').is_file()
for arm in ('aligned','zero'):
 m=json.loads(pathlib.Path(f'checkpoints/RDD-CSRA-prefix6-conditional1600-{arm}-L400K-20260913/complete.json').read_text())
 assert m['complete'] and m['step']==1600 and not m['smoke']
PY
check_memory() {
 mapfile -t mem < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#mem[@]} == 4 ]] || return 1
 for m in "${mem[@]}"; do ((m>=22528)) || return 1; done
}
until check_memory; do echo 'Waiting for22GiB free on0-3; other tasks untouched'; sleep 30; done
pids=()
for gpu in 0 1 2 3; do
 CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_fid_pilot.py sample \
  --root "$ROOT" --rank "$gpu" > "logs/csra_prefix6_fid10k_shard${gpu}_20260914.log" 2>&1 &
 pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
"$PY" 2d/attention_analysis/sample_csra_prefix_fid_pilot.py assemble --root "$ROOT"
# CPU-only ADM avoids competing for GPU memory; cache the reference across arms.
CUDA_VISIBLE_DEVICES='' TF_NUM_INTRAOP_THREADS=8 TF_NUM_INTEROP_THREADS=2 \
 "$PY" -u 2d/diagnostics/evaluate_many_npz.py --repo-root "$PWD" \
 --reference "$PWD/eval_results/reference/VIRTUAL_imagenet256_labeled.npz" \
 --sample "original_dense=$PWD/$ROOT/original_dense.npz" \
 --sample "aligned=$PWD/$ROOT/aligned.npz" --sample "zero=$PWD/$ROOT/zero.npz" \
 --output-dir "$ROOT/metrics" --name-suffix fid10k \
 2>&1 | tee logs/csra_prefix6_fid10k_metrics_20260914.log
echo 'Completed same-N IS/FID/sFID/Precision/Recall; this is NOT final50K FID.'
