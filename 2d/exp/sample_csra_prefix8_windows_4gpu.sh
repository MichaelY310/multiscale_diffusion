#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do (( value >= 18000 )) || return 1; done
}
until check_memory; do echo 'Waiting for18GiB/card on0–3, unrelated jobs untouched'; sleep 30; done
pids=()
seeds=(20260921 20275004 20260921 20275004)
windows=(45 45 60 60)
for gpu in 0 1 2 3; do
  seed="${seeds[$gpu]}"; window="${windows[$gpu]}"
  out="eval_results/csra_prefix8_window${window}_20260913/seed_${seed}"
  [[ ! -f "$out/metrics.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
    --prefix-depth 8 --checkpoint-template 'checkpoints/RDD-CSRA-prefix8-{mode}-L400K-20260913/best.pt' \
    --reuse-fine-steps "$window" --seed "$seed" --output-dir "$out" \
    > "logs/csra_prefix8_window${window}_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
(( failed == 0 )) || exit 1
echo 'Two windows, two development seeds, matched aligned/zero previews complete'
