#!/usr/bin/env python3
"""Deterministically reconstruct an ADM image NPZ through the local SD VAE."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from diffusers.models import AutoencoderKL


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--output-npz", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-images", type=int, default=10_000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.max_images <= 0:
        parser.error("batch-size and max-images must be positive")

    input_path = args.input_npz.expanduser().resolve()
    vae_path = args.vae_path.expanduser().resolve()
    output_path = args.output_npz.expanduser().resolve()
    if not input_path.is_file() or not (vae_path / "config.json").is_file():
        parser.error("input NPZ or local VAE is missing")
    archive = np.load(input_path, allow_pickle=False)
    if "arr_0" not in archive.files:
        parser.error("input NPZ must contain arr_0")
    images = archive["arr_0"]
    n = min(len(images), args.max_images)
    if images.ndim != 4 or images.shape[1:] != (256, 256, 3) or images.dtype != np.uint8:
        parser.error(f"expected uint8 [N,256,256,3], got {images.shape} {images.dtype}")

    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(str(vae_path), local_files_only=True).to(device).eval()
    reconstructed = np.empty((n, 256, 256, 3), dtype=np.uint8)
    with torch.inference_mode():
        for start in range(0, n, args.batch_size):
            end = min(n, start + args.batch_size)
            batch = torch.from_numpy(images[start:end]).to(device=device, non_blocking=True)
            batch = batch.permute(0, 3, 1, 2).float().div_(127.5).sub_(1.0)
            # Posterior mean makes the control deterministic and isolates the
            # VAE representation/decoder rather than adding sampling noise.
            latent = vae.encode(batch).latent_dist.mean.mul_(0.18215)
            decoded = vae.decode(latent / 0.18215).sample.clamp_(-1, 1)
            reconstructed[start:end] = (
                ((decoded + 1.0) * 127.5)
                .round_()
                .byte()
                .permute(0, 2, 3, 1)
                .cpu()
                .numpy()
            )
            print(f"{end:>6}/{n}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, arr_0=reconstructed)
    os.replace(temporary, output_path)
    print(f"[done] {output_path} {reconstructed.shape}", flush=True)


if __name__ == "__main__":
    main()
