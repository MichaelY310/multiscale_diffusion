#!/usr/bin/env python3
"""Download the official 512 SLat flow and convert it for BasicTrainer."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo", default="microsoft/TRELLIS.2-4B")
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stem = f"slat_flow_img2shape_dit_1_3B_{args.resolution}_bf16"
    filename = f"ckpts/{stem}.safetensors"
    source = hf_hub_download(args.repo, filename=filename)
    destination = args.output_dir / f"{stem}.pt"
    if not destination.exists():
        state = load_file(source, device="cpu")
        torch.save(state, destination)
    print(destination)


if __name__ == "__main__":
    main()
