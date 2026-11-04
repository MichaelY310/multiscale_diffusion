#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/direct_reuse_fullmodel_20260915
exec > >(tee -a "$R/preview_launcher.log") 2>&1
exec 9>"$R/preview.lock"
flock -n 9 || exit 1
for item in train_reuse6/complete.json train_dense/complete.json video_temporal221/metrics.json; do
  until [[ -f "$R/$item" ]]; do echo "Waiting for $item"; sleep 20; done
done
while true; do
  ok=1
  while IFS= read -r free; do ((free >= 22000)) || ok=0; done < <(
    nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  ((ok)) && break
  echo 'Waiting for22,000MiB free on GPUs0-3'; sleep 20
done
export OMP_NUM_THREADS=2
PY="$PWD/.venv-rdd-eval/bin/python"
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_direct_reuse_fullmodel.py \
    --root "$R" --rank "$gpu" --batch 16 > "$R/preview_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed == 0)) || exit 1
"$PY" 2d/attention_analysis/summarize_direct_reuse_fullmodel.py --root "$R"
echo 'COMPLETE audited image/video direct reuse pilot'
