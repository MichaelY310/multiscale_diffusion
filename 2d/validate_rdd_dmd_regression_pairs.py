#!/usr/bin/env python3
"""Decode a deterministic RDD-DMD pair subset and enforce basic quality gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers.models import AutoencoderKL
from torchvision.utils import save_image


def load_subset(root: Path, maximum: int) -> tuple[dict, torch.Tensor, torch.Tensor]:
    with open(root / "metadata.json", "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    latents = []
    labels = []
    for manifest_path in sorted(root.glob("manifest_rank*.jsonl")):
        with open(manifest_path, "r", encoding="utf-8") as manifest:
            for line in manifest:
                entry = json.loads(line)
                payload = torch.load(
                    root / entry["file"], map_location="cpu", weights_only=False
                )
                latents.append(payload["target_latent"].float())
                labels.append(payload["labels"].long())
                if sum(item.shape[0] for item in latents) >= maximum:
                    break
        if sum(item.shape[0] for item in latents) >= maximum:
            break
    if not latents:
        raise FileNotFoundError(f"no pair shards found in {root}")
    return metadata, torch.cat(latents)[:maximum], torch.cat(labels)[:maximum]


def main(args: argparse.Namespace) -> None:
    root = Path(args.pair_dir).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata, latents, labels = load_subset(root, args.max_images)
    if int(metadata.get("sampler_version", 0)) != 2:
        raise RuntimeError("quality gate requires sampler_version=2")
    if not bool(metadata.get("determinism_verified", False)):
        raise RuntimeError("pair builder did not record a determinism check")
    if not torch.isfinite(latents).all():
        raise RuntimeError("teacher target latents contain NaN or Inf")

    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True
    ).to(device).eval()
    for parameter in vae.parameters():
        parameter.requires_grad_(False)
    decoded_batches = []
    with torch.no_grad():
        for start in range(0, latents.shape[0], args.decode_batch):
            batch = latents[start : start + args.decode_batch].to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                decoded = vae.decode(batch / 0.18215).sample
            decoded_batches.append((decoded.float().cpu() / 2.0 + 0.5).clamp(0, 1))
    images = torch.cat(decoded_batches)
    save_image(images, output / "teacher_targets.png", nrow=args.grid_columns)

    flat = images.flatten(2)
    per_image_spatial_std = flat.std(dim=2).mean(dim=1)
    q05 = torch.quantile(flat, 0.05, dim=2)
    q95 = torch.quantile(flat, 0.95, dim=2)
    per_image_dynamic_range = (q95 - q05).mean(dim=1)
    metrics = {
        "pair_dir": str(root),
        "count": int(images.shape[0]),
        "labels": labels.tolist(),
        "latent_std": float(latents.std()),
        "decoded_spatial_std_mean": float(per_image_spatial_std.mean()),
        "decoded_spatial_std_median": float(per_image_spatial_std.median()),
        "decoded_dynamic_range_mean": float(per_image_dynamic_range.mean()),
        "decoded_dynamic_range_median": float(per_image_dynamic_range.median()),
        "minimum_latent_std": float(args.min_latent_std),
        "minimum_decoded_spatial_std_median": float(args.min_spatial_std),
        "sampler": metadata.get("sampler"),
        "sampler_version": metadata.get("sampler_version"),
        "determinism_verified": metadata.get("determinism_verified"),
    }
    metrics["passed"] = bool(
        metrics["latent_std"] >= args.min_latent_std
        and metrics["decoded_spatial_std_median"] >= args.min_spatial_std
    )
    with open(output / "quality_gate.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2), flush=True)
    if not metrics["passed"]:
        raise SystemExit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-images", type=int, default=16)
    parser.add_argument("--decode-batch", type=int, default=4)
    parser.add_argument("--grid-columns", type=int, default=4)
    parser.add_argument("--min-latent-std", type=float, default=0.5)
    parser.add_argument("--min-spatial-std", type=float, default=0.08)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
