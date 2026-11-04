#!/usr/bin/env bash
# Reproduce the final resident-DiT timing suite on physical GPUs 0-3.
# This launcher never replaces an existing tmux session.
set -euo pipefail

ROOT="${RDD_ROOT:-/scr/yuheng/DevOpsBench/multiscale_diffusion}"
PYTHON="${PYTHON:-${ROOT}/.venv-rdd-video/bin/python}"
OUT="${OUT:-${ROOT}/eval_results/rdd_fastwan_nearest_boundary_formal_repro}"
SUITE="rdd_video/experiments/fastwan_torch_nearest_boundary_suite.json"
FOX="A red fox walks slowly through a snowy forest while snow falls from the trees, cinematic camera movement."
CITY="A person in a yellow raincoat walks through a neon-lit city street at night, wet pavement reflections, cinematic camera movement."
COAST="A blue vintage car drives along a coastal road at sunset, ocean waves and cliffs in the background, cinematic tracking shot."

mkdir -p "${OUT}"

launch() {
  local session="$1" gpu="$2" prompt="$3" size="$4" warmup="$5" rounds="$6" port="$7"
  local run_dir="${OUT}/${session}"
  if tmux has-session -t "${session}" 2>/dev/null; then
    echo "Refusing to replace existing session: ${session}" >&2
    return 1
  fi
  mkdir -p "${run_dir}"
  tmux new-session -d -s "${session}" \
    "cd '${ROOT}' && CUDA_VISIBLE_DEVICES='${gpu}' '${PYTHON}' -u rdd_video/benchmark_fastwan_paired_suite.py --configs-json '${SUITE}' --baseline-name fixed_vsa20 --prompt '${prompt}' --size '${size}' --frame-num 29 --warmup-rounds '${warmup}' --timed-rounds '${rounds}' --output-dir '${run_dir}' --master-port '${port}' --device cuda:0 2>&1 | tee '${run_dir}/run.log'"
  echo "started ${session} on physical GPU ${gpu}"
}

launch rdd_resident_fox832_g0 0 "${FOX}" 832x448 5 50 29920
launch rdd_resident_city832_g1 1 "${CITY}" 832x448 5 50 29921
launch rdd_resident_coast832_g2 2 "${COAST}" 832x448 5 50 29922
launch rdd_resident_fox1280_g3 3 "${FOX}" 1280x704 3 30 29923

cat <<'EOF'
The 1664x896 scale check is intentionally separate because it needs about 20.6 GiB
of resident GPU memory. Run it only on an otherwise-idle GPU 0-3 with the same
suite, 3 warmups, and 20 timed rounds.
EOF
