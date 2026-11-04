"""
Upload multi-scale DiT cascade checkpoints to HuggingFace Hub.

This script strips the optimizer state (which is only needed to resume training)
and uploads the EMA weights + model config to a HuggingFace model repository.

The resulting repo will contain:
  checkpoints_32x32/ema.pt   — DiT-XL/2, input_size=32, t ∈ [0, 300)
  checkpoints_16x16/ema.pt   — DiT-L/2,  input_size=16, t ∈ [0, 600)
  checkpoints_8x8/ema.pt     — DiT-B/2,  input_size=8,  t ∈ [0, 1000)
  config.json                — model architecture params

Usage:
  huggingface-cli login        # do this once
  python scripts/upload_huggingface.py \
      --ckpt_dir 2d/results/013-DiT-B-2-multiscale-no-drift-v2 \
      --repo_id  your-username/multiscale-dit-cascade-imagenet256 \
      --step     68000

Loading saved weights later:
  from huggingface_hub import hf_hub_download
  import torch, sys, os

  # You still need the DiT model definition in your path:
  sys.path.insert(0, "path/to/scale_diffusion/DiT")
  from models import DiT_models

  def load_cascade(repo_id):
      configs = {
          "32x32": dict(arch="DiT-XL/2", input_size=32, folder="checkpoints_32x32"),
          "16x16": dict(arch="DiT-L/2",  input_size=16, folder="checkpoints_16x16"),
          "8x8":   dict(arch="DiT-B/2",  input_size=8,  folder="checkpoints_8x8"),
      }
      models = {}
      for name, cfg in configs.items():
          path = hf_hub_download(repo_id, f"{cfg['folder']}/ema.pt")
          state_dict = torch.load(path, map_location="cpu", weights_only=True)
          model = DiT_models[cfg["arch"]](input_size=cfg["input_size"], num_classes=1000)
          model.load_state_dict(state_dict)
          model.eval()
          models[name] = model
      return models
"""

import argparse
import json
import os
import sys
import tempfile

import torch
from huggingface_hub import HfApi, create_repo


MODEL_CONFIG = {
    "32x32": {
        "arch": "DiT-XL/2",
        "input_size": 32,
        "num_classes": 1000,
        "timestep_range": [0, 300],
        "folder": "checkpoints_32x32",
        "description": "Handles fine details (t ∈ [0, 300))",
    },
    "16x16": {
        "arch": "DiT-L/2",
        "input_size": 16,
        "num_classes": 1000,
        "timestep_range": [0, 600],
        "folder": "checkpoints_16x16",
        "description": "Handles mid-frequency structure (t ∈ [0, 600))",
    },
    "8x8": {
        "arch": "DiT-B/2",
        "input_size": 8,
        "num_classes": 1000,
        "timestep_range": [0, 1000],
        "folder": "checkpoints_8x8",
        "description": "Handles global structure (t ∈ [0, 1000))",
    },
}


def extract_ema(ckpt_path: str) -> dict:
    """Load a training checkpoint and return only the EMA state dict."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt["ema"]


def make_readme(repo_id: str, step: int) -> str:
    return f"""---
license: mit
tags:
  - diffusion
  - image-generation
  - imagenet
  - dit
---

# Multi-Scale DiT Cascade — ImageNet 256×256

A cascade of three class-conditional DiT models trained on ImageNet-1k-256×256
**without drift**, following a coarse-to-fine schedule:

| Stage | Architecture | Latent Size | Timestep Range | Role |
|-------|-------------|-------------|----------------|------|
| 1st   | DiT-B/2     | 8×8         | t ∈ [0, 1000)  | Global structure |
| 2nd   | DiT-L/2     | 16×16       | t ∈ [0, 600)   | Mid-frequency |
| 3rd   | DiT-XL/2   | 32×32       | t ∈ [0, 300)   | Fine details |

All models use a shared standard 1000-step DDPM schedule with the
`stabilityai/sd-vae-ft-ema` VAE (latent space). Checkpoint at step **{step:,}**.

## Files

```
checkpoints_8x8/ema.pt    — DiT-B/2,  EMA weights
checkpoints_16x16/ema.pt  — DiT-L/2,  EMA weights
checkpoints_32x32/ema.pt  — DiT-XL/2, EMA weights
config.json               — architecture parameters
```

## Loading

```python
from huggingface_hub import hf_hub_download
import torch, sys

# Clone https://github.com/your-username/scale_diffusion and add DiT to path
sys.path.insert(0, "scale_diffusion/DiT")
from models import DiT_models

def load_cascade(repo_id="{repo_id}", device="cuda"):
    configs = [
        dict(key="32x32", arch="DiT-XL/2", input_size=32, folder="checkpoints_32x32"),
        dict(key="16x16", arch="DiT-L/2",  input_size=16, folder="checkpoints_16x16"),
        dict(key="8x8",   arch="DiT-B/2",  input_size=8,  folder="checkpoints_8x8"),
    ]
    models = {{}}
    for cfg in configs:
        path = hf_hub_download(repo_id, f"{{cfg['folder']}}/ema.pt")
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
        model = DiT_models[cfg["arch"]](input_size=cfg["input_size"], num_classes=1000)
        model.load_state_dict(state_dict)
        model.eval().to(device)
        models[cfg["key"]] = model
    return models

models = load_cascade()
```

## Training

Models are trained jointly; each sees only its assigned timestep sub-range.
At inference, the cascade runs **8×8 → 16×16 → 32×32**, with bilinear
upsampling and re-noising between stages.
"""


def main():
    parser = argparse.ArgumentParser(description="Upload cascade DiT checkpoints to HuggingFace Hub")
    parser.add_argument("--ckpt_dir", required=True,
                        help="Path to experiment directory, e.g. 2d/results/013-...")
    parser.add_argument("--repo_id", required=True,
                        help="HuggingFace repo id, e.g. username/my-model")
    parser.add_argument("--step", type=int, required=True,
                        help="Checkpoint step number, e.g. 68000")
    parser.add_argument("--private", action="store_true",
                        help="Create a private repository")
    args = parser.parse_args()

    step_str = f"{args.step:07d}"
    api = HfApi()

    print(f"Creating repo: {args.repo_id} (private={args.private})")
    create_repo(args.repo_id, repo_type="model", exist_ok=True, private=args.private)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Upload config
        config_path = os.path.join(tmpdir, "config.json")
        with open(config_path, "w") as f:
            json.dump(MODEL_CONFIG, f, indent=2)
        api.upload_file(
            path_or_fileobj=config_path,
            path_in_repo="config.json",
            repo_id=args.repo_id,
        )
        print("Uploaded config.json")

        # Upload README
        readme_path = os.path.join(tmpdir, "README.md")
        with open(readme_path, "w") as f:
            f.write(make_readme(args.repo_id, args.step))
        api.upload_file(
            path_or_fileobj=readme_path,
            path_in_repo="README.md",
            repo_id=args.repo_id,
        )
        print("Uploaded README.md")

        # Extract and upload EMA weights for each stage
        for name, cfg in MODEL_CONFIG.items():
            src_path = os.path.join(args.ckpt_dir, cfg["folder"], f"{step_str}.pt")
            if not os.path.exists(src_path):
                print(f"  WARNING: {src_path} not found, skipping {name}")
                continue

            print(f"Extracting EMA from {src_path} ...")
            ema_state = extract_ema(src_path)

            ema_path = os.path.join(tmpdir, f"ema_{name}.pt")
            torch.save(ema_state, ema_path)

            size_mb = os.path.getsize(ema_path) / 1e6
            dest = f"{cfg['folder']}/ema.pt"
            print(f"  Uploading {dest} ({size_mb:.0f} MB) ...")
            api.upload_file(
                path_or_fileobj=ema_path,
                path_in_repo=dest,
                repo_id=args.repo_id,
            )
            print(f"  Done: {name}")

    print(f"\nAll done! View at: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
