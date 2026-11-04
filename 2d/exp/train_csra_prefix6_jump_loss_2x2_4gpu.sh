#!/usr/bin/env bash
# Matched old/jump loss x aligned/zero; all fixed parent1600 -> latest2400.
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
ROOT=eval_results/csra_prefix6_jump_loss_20260914
CAL="$ROOT/calibration.json"
export OMP_NUM_THREADS=4 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1
mkdir -p "$ROOT" logs
exec 9>eval_results/csra_prefix6_jump_loss_20260914.lock
flock -n 9 || { echo 'This exact experiment already runs'; exit 1; }
CUDA_VISIBLE_DEVICES='' "$PY" -m unittest discover -s tests -p 'test_csra_*.py'
CUDA_VISIBLE_DEVICES='' "$PY" 2d/attention_analysis/csra_jump_weighted_loss.py --output "$CAL"
check_memory() {
  mapfile -t values < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
  [[ ${#values[@]} == 4 ]] || return 1
  for value in "${values[@]}"; do ((value>=26624)) || return 1; done
}
wait_memory() {
  until check_memory; do echo 'Waiting for26GiB/card on0-3; never stop unrelated jobs'; sleep 30; done
}
disk_guard() {
  free_kb=$(df -Pk checkpoints | awk 'NR==2 {print $4}')
  ((free_kb>=50*1024*1024)) || { echo 'Need50GiB disk; no files deleted'; exit 1; }
}
verify_complete() {
  "$PY" - "$1" "$2" "$3" "$4" <<'PY'
import json,sys
p,loss,source,smoke=sys.argv[1:]
m=json.load(open(p));smoke=smoke=='yes'
assert m['complete'] and m['smoke']==smoke and m['step']==(1602 if smoke else 2400)
assert m['config']['coarse_source']==source and m['config']['prefix_depth']==6
e=m['experiment'];assert e['loss_mode']==loss and e['parent_step']==1600
assert e['per_gpu_batch']==64 and e['world_size']==4 and e['accumulation']==1
PY
}
disk_guard
for loss in old jump; do
  for source in aligned zero; do
    out="$ROOT/smoke_${loss}_${source}"
    if [[ -f "$out/complete.json" ]]; then verify_complete "$out/complete.json" "$loss" "$source" yes; continue; fi
    wait_memory
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_prefix6_jump_loss.py --coarse-source "$source" --loss-mode "$loss" \
      --calibration "$CAL" --extra-steps 2 --smoke --output-dir "$out" \
      2>&1 | tee -a "logs/csra_prefix6_jump_loss_smoke_${loss}_${source}_20260914.log"
    verify_complete "$out/complete.json" "$loss" "$source" yes
  done
done
# Confirm that all 4 arms, across all ranks, actually resumed identical data.
"$PY" - "$ROOT" <<'PY'
import json,pathlib,sys
r=pathlib.Path(sys.argv[1])
for rank in range(4):
 for step in (1601,1602):
  records=[json.load(open(r/f'smoke_{loss}_{src}'/f'start_trace_rank{rank}_step{step}.json'))
           for loss in ('old','jump') for src in ('aligned','zero')]
  for key in ('indices_sha256','times','epoch','batch_index'):
   assert all(row[key]==records[0][key] for row in records), key
print('All 4x64 real-optimizer smokes passed; matched resumed data verified on ALL ranks.')
PY
for loss in old jump; do
  for source in aligned zero; do
    out="checkpoints/RDD-CSRA-prefix6-jump-loss-${loss}-${source}-L400K-20260914"
    if [[ -f "$out/complete.json" ]]; then verify_complete "$out/complete.json" "$loss" "$source" no; continue; fi
    wait_memory; disk_guard
    resume=()
    if [[ -f "$out/latest.pt" ]]; then resume=(--resume "$out/latest.pt");
    elif [[ -f "$out/best.pt" ]]; then resume=(--resume "$out/best.pt"); fi
    CUDA_VISIBLE_DEVICES=0,1,2,3 "$PY" -m torch.distributed.run --standalone --nproc_per_node=4 \
      2d/attention_analysis/train_csra_prefix6_jump_loss.py --coarse-source "$source" --loss-mode "$loss" \
      --calibration "$CAL" --extra-steps 800 --output-dir "$out" "${resume[@]}" \
      2>&1 | tee -a "logs/csra_prefix6_jump_loss_${loss}_${source}_20260914.log"
    verify_complete "$out/complete.json" "$loss" "$source" no
  done
done
# SAME fixed256 development samples as previous prefix6 tests. Use latest2400,
# not validation-selected best. Evaluate only after every arm has finished.
for loss in old jump; do
  wait_memory
  pids=()
  for gpu in 0 1 2 3; do
    out="$ROOT/quality_${loss}"
    [[ ! -f "$out/shard_${gpu}/metrics.json" ]] || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/sample_csra_prefix_quality_pilot.py \
      --rank "$gpu" --prefix-depth 6 --checkpoint-step 2400 --seed-base 20460913 --memory-fraction .18 \
      --checkpoint-template "checkpoints/RDD-CSRA-prefix6-jump-loss-${loss}-{arm}-L400K-20260914/latest.pt" \
      --output-root "$out" > "logs/csra_prefix6_jump_loss_quality_${loss}_gpu${gpu}_20260914.log" 2>&1 &
    pids+=("$!")
  done
  failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  ((failed==0)) || exit 1
  "$PY" 2d/attention_analysis/summarize_csra_prefix_quality_pilot.py --root "$out" \
    --prefix-depth 6 --checkpoint-step 2400 --seed-base 20460913
done
echo 'Four fixed2400 endpoints and two paired256 quality pilots complete. Review before further training/FID.'
