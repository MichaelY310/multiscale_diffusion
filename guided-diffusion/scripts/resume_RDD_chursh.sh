cd /home/tiger/multiscale_diffusion/guided-diffusion
source .venv/bin/activate

export RESUME_DIR=/home/tiger/multiscale_diffusion/guided-diffusion/guided_diffusion/results_lsun_church_gpu0/004
export DATASET_PATH=/home/tiger/multiscale_diffusion/lsun/lsun_church_train
export DATASET_SPLIT=train
export BATCH_SIZE=8
export NUM_WORKERS=8
export EPOCHS=5
export WORLD_SIZE_FOR_TRAIN=1
export DIST_BACKEND=gloo
export CUDA_VISIBLE_DEVICES=0

export RESUME_STEP=$(
python - <<'PY'
import glob
import os
import re

resume_dir = os.environ["RESUME_DIR"]
ckpt_dirs = ["checkpoints_64", "checkpoints_128", "checkpoints_256"]

step_sets = []
for ckpt_dir in ckpt_dirs:
    steps = set()
    for path in glob.glob(os.path.join(resume_dir, ckpt_dir, "*.pt")):
        m = re.search(r"(\d+)\.pt$", os.path.basename(path))
        if m:
            steps.add(int(m.group(1)))
    step_sets.append(steps)

common = set.intersection(*step_sets)
if not common:
    raise SystemExit(f"No common checkpoint step found under {resume_dir}")

print(max(common))
PY
)

export STEPS_PER_EPOCH=$(
python - <<'PY'
import os
from datasets import load_dataset

dataset_path = os.environ["DATASET_PATH"]
split = os.environ.get("DATASET_SPLIT", "train")
batch_size = int(os.environ["BATCH_SIZE"])
world_size = int(os.environ.get("WORLD_SIZE_FOR_TRAIN", "1"))

ds = load_dataset(dataset_path, split=split)
print(len(ds) // (batch_size * world_size))
PY
)

export TOTAL_STEPS=$((RESUME_STEP + EPOCHS * STEPS_PER_EPOCH))

echo "RESUME_STEP=${RESUME_STEP}"
echo "STEPS_PER_EPOCH=${STEPS_PER_EPOCH}"
echo "EPOCHS=${EPOCHS}"
echo "TOTAL_STEPS=${TOTAL_STEPS}"

bash scripts/train_rdd_lsun_church_gpu0.sh