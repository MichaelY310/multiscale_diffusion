#!/usr/bin/env python3
"""Precompute one frozen DINOv2 CLS condition vector per rendered asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def first_rgba(render_root: Path, asset: str) -> Image.Image:
    root = render_root / asset
    metadata = json.loads((root / "transforms.json").read_text(encoding="utf-8"))
    return Image.open(root / metadata["frames"][0]["file_path"]).convert("RGBA")


def crop_and_normalize(image: Image.Image, size: int) -> torch.Tensor:
    rgba = np.asarray(image).astype(np.float32) / 255.0
    alpha = rgba[..., 3]
    nonzero = np.nonzero(alpha > 0)
    if not len(nonzero[0]):
        raise ValueError("render has an empty alpha mask")
    x0, x1 = int(nonzero[1].min()), int(nonzero[1].max())
    y0, y1 = int(nonzero[0].min()), int(nonzero[0].max())
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    half = max(x1 - x0, y1 - y0) * 0.55
    image = image.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    rgba = np.asarray(image).astype(np.float32) / 255.0
    rgb = rgba[..., :3] * rgba[..., 3:4]
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
    return (tensor - mean) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    files = sorted(args.latent_dir.glob("*.npz"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no latents under {args.latent_dir}")
    args.output.mkdir(parents=True, exist_ok=True)
    model = torch.hub.load("facebookresearch/dinov2", args.model).eval().to(args.device)
    for index, path in enumerate(files, start=1):
        output = args.output / f"{path.stem}.npy"
        if output.exists():
            continue
        image = crop_and_normalize(
            first_rgba(args.render_dir, path.stem), args.image_size
        ).unsqueeze(0).to(args.device)
        with torch.inference_mode(), torch.autocast(
            device_type=torch.device(args.device).type,
            dtype=torch.bfloat16,
            enabled=torch.device(args.device).type == "cuda",
        ):
            features = model.forward_features(image)["x_norm_clstoken"]
        np.save(output, features[0].float().cpu().numpy())
        print(f"[{index}/{len(files)}] {path.stem}", flush=True)


if __name__ == "__main__":
    main()
