#!/usr/bin/env bash
set -euo pipefail
cd /scr/yuheng/DevOpsBench/multiscale_diffusion
PY=.venv-rdd-eval/bin/python
ROOT=eval_results/csra_state_replay_20260914
mkdir -p "$ROOT"
exec 9>"$ROOT/collect.lock"
flock -n 9 || { echo 'Exact replay collector already running'; exit 1; }
export OMP_NUM_THREADS=2
CUDA_VISIBLE_DEVICES=0 "$PY" -m unittest discover -s tests -p test_csra_state_replay.py -v 2>&1 | tee "$ROOT/unit_tests.log"
wait_memory() {
    while true; do
        mapfile -t available < <(nvidia-smi -i 0,1,2,3 --query-gpu=memory.free --format=csv,noheader,nounits)
        ready=1; [[ ${#available[@]} == 4 ]] || ready=0
        for n in "${available[@]}"; do ((n>=20000)) || ready=0; done
        ((ready)) && break
        echo 'Waiting for20GiB on0-3; other processes untouched'; sleep 30
    done
}
wait_memory
if [[ ! -f "$ROOT/smoke_data/shard_0/manifest.json" ]]; then
    CUDA_VISIBLE_DEVICES=0 "$PY" -u 2d/attention_analysis/collect_csra_state_replay.py --shard 0 --smoke \
        2>&1 | tee "$ROOT/collect_smoke.log"
fi
wait_memory
pids=()
for gpu in 0 1 2 3; do
    [[ ! -f "$ROOT/data/shard_$gpu/manifest.json" ]] || continue
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u 2d/attention_analysis/collect_csra_state_replay.py --shard "$gpu" \
        > "$ROOT/collect_gpu${gpu}.log" 2>&1 &
    pids+=("$!")
done
failed=0; for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
((failed==0)) || exit 1
echo 'Matched4096train+128validation teacher/student snapshot pairs complete. Training is a separate checked step.'
