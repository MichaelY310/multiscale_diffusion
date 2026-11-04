#!/usr/bin/env bash
# Four matched arms; each adds800updates to its fixed3200parent.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
exec 9>eval_results/csra_prefix8_hidden_2x2_20260913.lock
flock -n 9 || { echo 'This exact2x2 experiment is already running'; exit 1; }
"$PY" -m unittest discover -s tests -p 'test_csra_*.py'
free_kb=$(df -Pk checkpoints | awk 'NR==2 {print $4}')
((free_kb>=45*1024*1024)) || { echo 'Need45GiB disk; nothing deleted'; exit 1; }
check_memory() {
 mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
 [[ ${#values[@]} == 4 ]] || return 1
 for value in "${values[@]}"; do ((value>=26500)) || return 1; done
}
wait_memory() { until check_memory; do echo 'Waiting for26GiB/card on0–3; other jobs untouched'; sleep 30; done; }
# Real4x64 optimizer smoke for every arm before any long job. It changes only
# ephemeral model/optimizer copies, never overwrites the parent/checkpoints.
for aux in on off; do
 for source in aligned zero; do
  smoke="eval_results/csra_prefix8_hidden_smoke_20260913/${aux}_${source}"
  if [[ -f "$smoke/complete.json" ]]; then
   "$PY" - "$smoke/complete.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));assert m['complete'] and m['smoke'] and m['step']==3202
PY
   continue
  fi
  wait_memory
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
   2d/attention_analysis/train_csra_prefix_hidden_pair.py --coarse-source "$source" --aux "$aux" \
   --extra-steps 2 --smoke --output-dir "$smoke" \
   2>&1 | tee -a "logs/csra_prefix8_hidden_smoke_${aux}_${source}_20260913.log"
 done
done
for aux in on off; do
 for source in aligned zero; do
  out="checkpoints/RDD-CSRA-prefix8-hidden-${aux}-${source}-L400K-20260913"
  if [[ -f "$out/complete.json" ]]; then
   "$PY" - "$out/complete.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1]));assert m['complete'] and not m['smoke'] and m['step']==4000
PY
   continue
  fi
  wait_memory
  resume=()
  if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
  elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
  CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
   2d/attention_analysis/train_csra_prefix_hidden_pair.py --coarse-source "$source" --aux "$aux" \
   --extra-steps 800 --output-dir "$out" "${resume[@]}" \
   2>&1 | tee -a "logs/csra_prefix8_hidden_${aux}_${source}_20260913.log"
 done
done
for aux in on off; do
 wait_memory
 pids=();seeds=(20260921 20260923 20275004)
 for gpu in 0 1 2; do
  seed="${seeds[$gpu]}";out="eval_results/csra_prefix8_hidden_${aux}_pair_20260913/seed_${seed}"
  [[ ! -f "$out/metrics.json" ]] || continue
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_preview.py \
   --prefix-depth 8 --seed "$seed" --output-dir "$out" \
   --checkpoint-template "checkpoints/RDD-CSRA-prefix8-hidden-${aux}-{mode}-L400K-20260913/best.pt" \
   > "logs/csra_prefix8_hidden_${aux}_preview_${seed}_20260913.log" 2>&1 &
  pids+=("$!")
 done
 failed=0
 for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
 ((failed==0)) || exit 1
done
echo '2x2hidden-supervision experiment and6paired previews complete'
