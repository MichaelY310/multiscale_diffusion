"""Download only the FastWan2.1 1.3B DMD transformer and scheduler.

The original Wan text encoder/tokenizer/VAE already stored beside this project
are reused by the exploratory RDD sampler, so downloading the full 29.2 GB
Diffusers repository would duplicate roughly 23 GB of unchanged components.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="FastVideo/FastWan2.1-T2V-1.3B-Diffusers",
    )
    parser.add_argument(
        "--output-dir",
        default="pretrained_models/FastWan2.1-T2V-1.3B-Diffusers",
    )
    args = parser.parse_args()

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=output,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "transformer/config.json",
            "transformer/*.safetensors",
        ],
    )
    print(f"FastWan minimal checkpoint ready: {path}")


if __name__ == "__main__":
    main()
