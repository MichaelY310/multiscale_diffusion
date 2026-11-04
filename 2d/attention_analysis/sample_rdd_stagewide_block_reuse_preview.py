#!/usr/bin/env python3
"""Small real DDPM preview for stage-wide RDD block-residual reuse."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "2d"))
from sample_fid_unified import _sample_batch_unified  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from probe_rdd_vs_dit_block_reuse import (  # noqa: E402
    load_rdd,
    manual_forward,
    upsample_tokens,
)

sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


class StageWideReuseModel:
    """Cache the final middle-stage updates and reuse selected fine blocks."""

    def __init__(self, model, layers: list[int], boundary_only: bool):
        self.model = model
        self.layers = layers
        self.boundary_only = boundary_only
        self.cached: dict[int, torch.Tensor] = {}
        self.fine_calls = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        if resolution == 16 and int(t[0].item()) == 300:
            output, trace = manual_forward(
                self.model, x, t, y, multires=True, capture=True
            )
            self.cached = {
                layer: upsample_tokens(trace[layer]["residual"], 16, "nearest")
                for layer in self.layers
            }
            return output
        if resolution == 32 and self.cached:
            self.fine_calls += 1
            if not self.boundary_only or self.fine_calls == 1:
                output, _ = manual_forward(
                    self.model,
                    x,
                    t,
                    y,
                    multires=True,
                    replacements=self.cached,
                    replacement_mode="residual",
                )
                return output
        output, _ = manual_forward(self.model, x, t, y, multires=True)
        return output


class ManualBaselineModel:
    """Run the exact same manual/autocast path without replacing any block."""

    def __init__(self, model):
        self.model = model

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        output, _ = manual_forward(self.model, x, t, y, multires=True)
        return output


def make_diffusions(num_steps: list[int]):
    resolutions = [32, 16, 8]
    boundaries = [0, 300, 600, 1000]
    diffusions = []
    for index, resolution in enumerate(resolutions):
        sampling_position = 2 - index
        diffusions.append(
            create_diffusion_with_drift(
                str(num_steps[sampling_position]),
                c=index if index < 2 else None,
                T_c=boundaries[index],
                T_c_plus_1=boundaries[index + 1],
                sample_scale=2,
            )
        )
    return diffusions


@torch.inference_mode()
def sample_once(model, diffusions, batch_size, num_steps, device, seed):
    torch.manual_seed(seed)
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    latents = _sample_batch_unified(
        batch_size,
        diffusions,
        [32, 16, 8],
        model,
        1000,
        False,
        1.0,
        "ddpm",
        "noise_filling",
        num_steps,
        device,
        torch.float32,
    )
    torch.cuda.synchronize(device)
    return latents, time.perf_counter() - start


@torch.inference_mode()
def decode(vae, latents):
    images = vae.decode(latents / 0.18215).sample
    return ((images.clamp(-1, 1) + 1) / 2).cpu()


def to_pil(tensor):
    array = (tensor.permute(1, 2, 0).numpy() * 255.0).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_grid(columns, output):
    tile = columns[0][1].shape[-1]
    rows = columns[0][1].shape[0]
    header = 38
    canvas = Image.new("RGB", (len(columns) * tile, header + rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, images) in enumerate(columns):
        x = column * tile
        draw.text((x + 5, 12), name, fill="black")
        for row in range(rows):
            canvas.paste(to_pil(images[row]), (x, header + row * tile))
    canvas.save(output)


def relative_l2(candidate, reference):
    error = (candidate.float() - reference.float()).flatten(1)
    denominator = reference.float().flatten(1)
    return float(
        (error.square().sum(1) / denominator.square().sum(1).clamp_min(1e-12))
        .sqrt()
        .mean()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-steps", default="80,80,90")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    num_steps = [int(value) for value in args.num_steps.split(",")]
    diffusions = make_diffusions(num_steps)
    base_model = load_rdd(Path(args.rdd_ckpt), device, torch.float32)

    settings = [
        ("baseline", ManualBaselineModel(base_model)),
        ("boundary3", StageWideReuseModel(base_model, [14, 17, 20], True)),
        ("stagewide3", StageWideReuseModel(base_model, [14, 17, 20], False)),
        ("stagewide4", StageWideReuseModel(base_model, [11, 14, 17, 20], False)),
    ]
    latents = {}
    timings = {}
    for name, wrapper in settings:
        sample, elapsed = sample_once(
            wrapper, diffusions, args.batch_size, num_steps, device, args.seed
        )
        latents[name] = sample
        timings[name] = elapsed

    vae = AutoencoderKL.from_pretrained(str(args.vae_path), local_files_only=True)
    vae.to(device).eval().requires_grad_(False)
    decoded = {name: decode(vae, value) for name, value in latents.items()}
    save_grid(
        [(name, decoded[name]) for name, _ in settings],
        output_dir / "comparison_grid.png",
    )
    baseline_time = timings["baseline"]
    report = {
        "checkpoint": str(args.rdd_ckpt),
        "sampler": "DDPM",
        "num_steps_coarse_middle_fine": num_steps,
        "upsample": "canonical noise_filling",
        "cfg": 1.0,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "settings": {},
    }
    for name, _ in settings:
        report["settings"][name] = {
            "seconds": timings[name],
            "speedup_vs_baseline": baseline_time / timings[name],
            "latency_reduction_percent": 100.0 * (1.0 - timings[name] / baseline_time),
            "latent_relative_l2_vs_baseline": relative_l2(latents[name], latents["baseline"]),
        }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
