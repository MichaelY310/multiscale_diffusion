#!/usr/bin/env python3
"""Generate one shared TRELLIS.2 structure support per held-out image.

RDD and flat geometry evaluation consume these exact same coordinate files.
Keeping the structure stage outside both geometry jobs prevents seed drift and
makes the scope of the speed/quality claim explicit.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from trellis2.pipelines import Trellis2ImageTo3DPipeline


def first_render(root: Path, asset: str) -> Image.Image:
    metadata = json.loads((root / asset / "transforms.json").read_text(encoding="utf-8"))
    return Image.open(root / asset / metadata["frames"][0]["file_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--guidance-strength", type=float, default=7.5)
    parser.add_argument("--model", default="microsoft/TRELLIS.2-4B")
    args = parser.parse_args()

    files = sorted(args.latents.glob("*.npz"))[: args.limit]
    if not files:
        raise SystemExit(f"no latent files under {args.latents}")
    args.output.mkdir(parents=True, exist_ok=True)

    pipeline = Trellis2ImageTo3DPipeline.from_pretrained(args.model)
    pipeline.cuda()
    latent_resolution = args.resolution // 16
    rows = []
    for index, path in enumerate(files):
        image = pipeline.preprocess_image(first_render(args.renders, path.stem))
        torch.manual_seed(args.seed + index)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        cond = pipeline.get_cond([image], 512)
        coords = pipeline.sample_sparse_structure(
            cond,
            latent_resolution,
            num_samples=1,
            sampler_params={
                "steps": args.steps,
                "guidance_strength": args.guidance_strength,
            },
        )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        coords_np = coords.detach().cpu().numpy().astype(np.int32)
        np.savez_compressed(args.output / path.name, coords=coords_np)
        rows.append(
            {
                "asset": path.stem,
                "tokens": len(coords_np),
                "seconds_including_condition": elapsed,
                "peak_memory_gib": torch.cuda.max_memory_allocated() / 2**30,
            }
        )

    manifest = {
        "model": args.model,
        "output_resolution": args.resolution,
        "latent_resolution": latent_resolution,
        "seed": args.seed,
        "structure_steps": args.steps,
        "guidance_strength": args.guidance_strength,
        "samples": rows,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
