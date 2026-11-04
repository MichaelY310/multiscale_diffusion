#!/usr/bin/env bash
# Separate CPU ADM environment and official OpenAI reference assets.
set -euo pipefail
EXPERIMENT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="$(cd "$EXPERIMENT_DIR/../../.." && pwd -P)"
ENV_DIR="${ADM_ENV_DIR:-$PROJECT_ROOT/.venv-adm}"
ASSET_DIR="${ADM_ASSET_DIR:-$PROJECT_ROOT/eval_results/reference}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
"$BOOTSTRAP_PYTHON" -c 'import sys; assert (3,10) <= sys.version_info[:2] <= (3,12), "Use Python3.10-3.12"'
if [[ ! -x "$ENV_DIR/bin/python" ]]; then "$BOOTSTRAP_PYTHON" -m venv "$ENV_DIR"; fi
export PIP_NO_CACHE_DIR=1
"$ENV_DIR/bin/python" -m pip install 'tensorflow-cpu==2.17.1' 'numpy==1.26.4' \
  'scipy==1.12.0' 'Pillow==11.3.0' 'requests==2.32.3' 'tqdm==4.67.1'
"$ENV_DIR/bin/python" -m pip check
mkdir -p "$ASSET_DIR"
BASE_URL=https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches
for NAME in classify_image_graph_def.pb VIRTUAL_imagenet256_labeled.npz; do
  URL="$BASE_URL/$NAME"
  if [[ "$NAME" == VIRTUAL_imagenet256_labeled.npz ]]; then URL="$BASE_URL/imagenet/256/$NAME"; fi
  if [[ ! -f "$ASSET_DIR/$NAME" ]]; then
    curl --fail --location --retry 5 --continue-at - --output "$ASSET_DIR/$NAME.part" "$URL"
    mv -- "$ASSET_DIR/$NAME.part" "$ASSET_DIR/$NAME"
  fi
done
"$ENV_DIR/bin/python" - "$ASSET_DIR" <<'PY'
import base64,hashlib,json,sys
from pathlib import Path
root=Path(sys.argv[1]);records=[]
# Content-Length/Content-MD5 from the official endpoints, checked2026-11-04.
for name,size,md5 in [('classify_image_graph_def.pb',95673916,'5rsVToX11DMcIqvN213PMQ=='),
                     ('VIRTUAL_imagenet256_labeled.npz',2037122530,'FMWRPh0k6IJ8G6Ydj8WuuQ==')]:
    path=root/name
    if path.stat().st_size!=size:raise ValueError(f'Wrong size: {path}')
    m=hashlib.md5();s=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):m.update(block);s.update(block)
    if base64.b64encode(m.digest()).decode()!=md5:raise ValueError(f'Official Content-MD5 mismatch: {path}')
    records.append({'file':name,'bytes':size,'sha256':s.hexdigest(),'official_content_md5':md5})
(root/'ADM_ASSETS.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(records,indent=2))
PY
CUDA_VISIBLE_DEVICES='' "$ENV_DIR/bin/python" -c 'import tensorflow.compat.v1 as tf; import numpy,scipy; print("ADM CPU environment ready",tf.__version__)'
