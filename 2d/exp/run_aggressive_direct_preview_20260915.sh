#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
R=/home/yuheng/rdd_experiments/aggressive_direct_reuse_20260915
mkdir -p "$R"
exec > >(tee -a "$R/preview_launcher.log") 2>&1
exec 7>"$R/preview.lock"
flock -n 7 || exit 1
for mask in reuse12 reuse18; do
  until [[ -f "$R/train_$mask/complete.json" ]]; do
    echo "Waiting for $mask training completion"; sleep 30
  done
done
while true; do
  ok=1
  while IFS= read -r free; do ((free>=26000)) || ok=0; done < <(
    nvidia-smi -i 0,1 --query-gpu=memory.free --format=csv,noheader,nounits)
  ((ok)) && break
  echo 'Waiting for26,000MiB free on GPU0/1'; sleep 30
done
export OMP_NUM_THREADS=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$PWD/.venv-rdd-eval/bin/python"
pids=()
for gpu in 0 1; do
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_aggressive_direct_reuse.py \
    --root "$R" --rank "$gpu" --batch 16 > "$R/preview_$gpu.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'COMPLETE32 paired samples,7 variants,actual sampling+VAE timing'
