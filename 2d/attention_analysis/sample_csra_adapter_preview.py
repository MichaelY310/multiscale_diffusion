#!/usr/bin/env python3
"""End-to-end rollout preview for trained RDD/DiT residual adapters.

This deliberately evaluates small, paired batches before any expensive FID run.
For RDD, the final 16x16-stage block residual is cached and the adapter predicts
selected 32x32-stage residuals.  For ordinary DiT, the matched control carries
the selected full-resolution residual from the previous NFE.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageDraw

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "2d"))
sys.path.insert(0, str(REPO_ROOT))

from probe_rdd_vs_dit_block_reuse import manual_forward, upsample_tokens  # noqa: E402
from sample_rdd_stagewide_block_reuse_preview import make_diffusions  # noqa: E402
from train_rdd_csra_adapter import (  # noqa: E402
    CSRAAdapterBank,
    load_frozen_teacher,
)
from csra_runtime import predict_adapter  # noqa: E402
from sample_fid_unified import _sample_batch_unified  # noqa: E402
from diffusion import create_diffusion  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


def load_adapter(path: Path, device: torch.device) -> tuple[CSRAAdapterBank, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    adapter = CSRAAdapterBank(
        list(checkpoint["layers"]),
        int(checkpoint["hidden_size"]),
        int(checkpoint["rank"]),
        architecture=checkpoint.get("adapter_architecture", "legacy_v1"),
    )
    adapter.load_state_dict(checkpoint["adapter"])
    return adapter.to(device).eval().requires_grad_(False), checkpoint


class RDDRollout:
    def __init__(
        self,
        model: torch.nn.Module,
        adapter: CSRAAdapterBank | None,
        layers: list[int],
        mode: str,
        adapter_precision: str = "legacy_autocast",
    ) -> None:
        if mode not in {"dense", "raw", "adapter"}:
            raise ValueError(mode)
        self.model = model
        self.adapter = adapter
        self.layers = layers
        self.mode = mode
        self.adapter_precision = adapter_precision
        self.coarse: dict[int, torch.Tensor] = {}
        self.replaced_blocks = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        resolution = x.shape[-1]
        if resolution == 16 and int(t[0].item()) == 300 and self.mode != "dense":
            output, trace = manual_forward(
                self.model, x, t, y, multires=True, capture_layers=set(self.layers)
            )
            self.coarse = {
                layer: upsample_tokens(trace[layer]["residual"], 16, "nearest")
                for layer in self.layers
            }
            return output

        if resolution == 32 and self.coarse and self.mode != "dense":
            if self.mode == "raw":
                replacements: dict[int, torch.Tensor | Callable] = self.coarse
            else:
                assert self.adapter is not None
                replacements = {
                    layer: (
                        lambda hidden, condition, layer=layer: predict_adapter(
                            self.adapter.adapters[str(layer)], self.coarse[layer],
                            hidden, t, self.adapter_precision, condition=condition,
                        )
                    )
                    for layer in self.layers
                }
            self.replaced_blocks += len(self.layers)
            output, _ = manual_forward(
                self.model,
                x,
                t,
                y,
                multires=True,
                replacements=replacements,
                replacement_mode="residual",
                conditioned_replacements=self.mode == "adapter",
            )
            return output

        output, _ = manual_forward(self.model, x, t, y, multires=True)
        return output


class DiTRollout:
    def __init__(
        self,
        model: torch.nn.Module,
        adapter: CSRAAdapterBank | None,
        layers: list[int],
        mode: str,
        refresh_period: int = 0,
        active_max_t: int = 299,
        adapter_precision: str = "legacy_autocast",
    ) -> None:
        if mode not in {"dense", "raw", "adapter"}:
            raise ValueError(mode)
        self.model = model
        self.adapter = adapter
        self.layers = layers
        self.mode = mode
        self.refresh_period = refresh_period
        self.active_max_t = active_max_t
        self.adapter_precision = adapter_precision
        self.cache: dict[int, torch.Tensor] = {}
        self.calls = 0
        self.replaced_blocks = 0

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        active = int(t[0].item()) <= self.active_max_t
        if not active:
            # The control adapter was deliberately trained only on the same
            # global t=0..299 interval occupied by RDD's 32x32 stage.  Keep the
            # ordinary DiT dense outside that matched interval, while retaining
            # the last exact residual as the source for the first active NFE.
            output, trace = manual_forward(
                self.model, x, t, y, multires=False, capture_layers=set(self.layers)
            )
            if self.mode != "dense":
                self.cache = {layer: trace[layer]["residual"] for layer in self.layers}
            return output

        refresh = (
            self.mode == "dense"
            or not self.cache
            or (self.refresh_period > 0 and self.calls % self.refresh_period == 0)
        )
        self.calls += 1
        if refresh:
            output, trace = manual_forward(
                self.model, x, t, y, multires=False, capture_layers=set(self.layers)
            )
            if self.mode != "dense":
                self.cache = {layer: trace[layer]["residual"] for layer in self.layers}
            return output

        if self.mode == "raw":
            replacements: dict[int, torch.Tensor | Callable] = self.cache
        else:
            assert self.adapter is not None
            replacements = {
                layer: (
                    lambda hidden, condition, layer=layer: predict_adapter(
                        self.adapter.adapters[str(layer)], self.cache[layer],
                        hidden, t, self.adapter_precision, condition=condition,
                    )
                )
                for layer in self.layers
            }
        output, trace = manual_forward(
            self.model,
            x,
            t,
            y,
            multires=False,
            replacements=replacements,
            replacement_mode="residual",
            conditioned_replacements=self.mode == "adapter",
            capture_layers=set(self.layers),
        )
        # The generic control has no persistent coarse-stage feature.  Carrying
        # the predicted residual is therefore the only stage-wide, no-refresh
        # analogue of RDD-CSRA and exposes recurrent error accumulation.
        self.cache = {layer: trace[layer]["residual"] for layer in self.layers}
        self.replaced_blocks += len(self.layers)
        return output


def synchronize(device: torch.device) -> None:
    torch.cuda.synchronize(device)


@torch.inference_mode()
def sample_rdd(
    wrapper: RDDRollout,
    batch_size: int,
    steps: list[int],
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, float]:
    torch.manual_seed(seed)
    synchronize(device)
    start = time.perf_counter()
    latent = _sample_batch_unified(
        batch_size,
        make_diffusions(steps),
        [32, 16, 8],
        wrapper,
        1000,
        False,
        1.0,
        "ddpm",
        "noise_filling",
        steps,
        device,
        torch.bfloat16,
    )
    synchronize(device)
    return latent, time.perf_counter() - start


@torch.inference_mode()
def sample_dit(
    wrapper: DiTRollout,
    batch_size: int,
    steps: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, float]:
    torch.manual_seed(seed)
    labels = torch.randint(0, 1000, (batch_size,), device=device)
    latent = torch.randn(batch_size, 4, 32, 32, device=device)
    diffusion = create_diffusion(str(steps))
    synchronize(device)
    start = time.perf_counter()
    result = diffusion.p_sample_loop(
        wrapper.forward,
        latent.shape,
        latent,
        clip_denoised=False,
        model_kwargs={"y": labels},
        progress=False,
        device=device,
    )
    synchronize(device)
    return result.float(), time.perf_counter() - start


@torch.inference_mode()
def decode(vae: AutoencoderKL, latent: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        image = vae.decode(latent.to(dtype=torch.bfloat16) / 0.18215).sample
    return ((image.float().clamp(-1, 1) + 1) / 2).cpu()


def relative_l2(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    error = (candidate.float() - reference.float()).flatten(1)
    denominator = reference.float().flatten(1)
    return float(
        (error.square().sum(1) / denominator.square().sum(1).clamp_min(1e-12))
        .sqrt()
        .mean()
    )


def to_pil(tensor: torch.Tensor) -> Image.Image:
    array = (tensor.permute(1, 2, 0).numpy() * 255).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(array)


def save_grid(columns: list[tuple[str, torch.Tensor]], output: Path) -> None:
    tile = int(columns[0][1].shape[-1])
    rows = int(columns[0][1].shape[0])
    header = 48
    canvas = Image.new("RGB", (len(columns) * tile, header + rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (name, images) in enumerate(columns):
        left = column * tile
        draw.text((left + 5, 8), name, fill="black")
        for row in range(rows):
            canvas.paste(to_pil(images[row]), (left, header + row * tile))
    canvas.save(output)


def parse_layers(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=["rdd", "dit"], required=True)
    parser.add_argument("--base-ckpt", required=True)
    parser.add_argument("--adapter-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--rdd-steps", default="80,80,90")
    parser.add_argument("--dit-steps", type=int, default=250)
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    parser.add_argument("--adapter-precision", choices=["legacy_autocast", "fp32"],
                        default="legacy_autocast")
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter, adapter_checkpoint = load_adapter(Path(args.adapter_ckpt), device)
    all_layers = list(adapter_checkpoint["layers"])
    early = all_layers[:3]
    late = all_layers[3:]
    model = load_frozen_teacher(Path(args.base_ckpt), device, "DiT-L/2", args.family)

    if args.family == "rdd":
        constructors = {
            "dense": lambda: RDDRollout(model, None, [], "dense"),
            "raw_all6": lambda: RDDRollout(model, None, all_layers, "raw"),
            "csra_early3": lambda: RDDRollout(model, adapter, early, "adapter", args.adapter_precision),
            "csra_late3": lambda: RDDRollout(model, adapter, late, "adapter", args.adapter_precision),
            "csra_all6": lambda: RDDRollout(model, adapter, all_layers, "adapter", args.adapter_precision),
        }
        steps = [int(value) for value in args.rdd_steps.split(",")]
        sampler = lambda wrapper, seed: sample_rdd(
            wrapper, args.batch_size, steps, device, seed
        )
        sampler_description = {"sampler": "DDPM", "stage_steps": steps, "upsample": "noise_filling"}
    else:
        constructors = {
            "dense": lambda: DiTRollout(model, None, [], "dense"),
            "raw_recurrent_all6": lambda: DiTRollout(model, None, all_layers, "raw"),
            "adapter_recurrent_early3": lambda: DiTRollout(model, adapter, early, "adapter", adapter_precision=args.adapter_precision),
            "adapter_recurrent_late3": lambda: DiTRollout(model, adapter, late, "adapter", adapter_precision=args.adapter_precision),
            "adapter_recurrent_all6": lambda: DiTRollout(model, adapter, all_layers, "adapter", adapter_precision=args.adapter_precision),
            "adapter_period4_all6": lambda: DiTRollout(model, adapter, all_layers, "adapter", 4, adapter_precision=args.adapter_precision),
        }
        sampler = lambda wrapper, seed: sample_dit(
            wrapper, args.batch_size, args.dit_steps, device, seed
        )
        sampler_description = {"sampler": "DDPM", "steps": args.dit_steps}

    latents: dict[str, torch.Tensor] = {}
    first_times: dict[str, float] = {}
    wrappers: dict[str, RDDRollout | DiTRollout] = {}
    for name, constructor in constructors.items():
        wrappers[name] = constructor()
        latents[name], first_times[name] = sampler(wrappers[name], args.seed)

    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval().requires_grad_(False)
    decoded = {name: decode(vae, latent) for name, latent in latents.items()}
    # Preserve historical result-directory keys but display actual layer counts.
    grid_names = {
        "raw_all6": f"raw ({len(all_layers)} layers)",
        "csra_early3": f"adapter ({len(early)} layers: {early})",
        "csra_late3": f"adapter ({len(late)} layers: {late})",
        "csra_all6": f"adapter ({len(all_layers)} layers)",
    }
    save_grid([(grid_names.get(name, name), decoded[name]) for name in constructors], output_dir / "comparison_grid.png")
    for name, images in decoded.items():
        arm_dir = output_dir / name
        arm_dir.mkdir(exist_ok=True)
        for index, image in enumerate(images):
            to_pil(image).save(arm_dir / f"{index:03d}.png")

    timing: dict[str, list[float]] = {name: [] for name in constructors}
    for repeat in range(args.benchmark_repeats):
        names = list(constructors)
        if repeat % 2:
            names.reverse()
        for name in names:
            _, elapsed = sampler(constructors[name](), args.seed + 1000 + repeat)
            timing[name].append(elapsed)

    dense_time = statistics.median(timing["dense"]) if timing["dense"] else first_times["dense"]
    report = {
        "family": args.family,
        "base_checkpoint": args.base_ckpt,
        "adapter_checkpoint": args.adapter_ckpt,
        "adapter_train_step": int(adapter_checkpoint["step"]),
        "adapter_architecture": adapter_checkpoint.get("adapter_architecture", "legacy_v1"),
        "adapter_precision": args.adapter_precision,
        "cfg_scale": 1.0,
        "layers": all_layers,
        "batch_size": args.batch_size,
        "seed": args.seed,
        **sampler_description,
        "settings": {},
    }
    for name in constructors:
        seconds = statistics.median(timing[name]) if timing[name] else first_times[name]
        report["settings"][name] = {
            "actual_layers": list(wrappers[name].layers),
            "first_seconds": first_times[name],
            "benchmark_seconds": timing[name],
            "median_seconds": seconds,
            "speedup_vs_dense": dense_time / seconds,
            "latency_reduction_percent": 100.0 * (1.0 - seconds / dense_time),
            "latent_relative_l2_vs_dense": relative_l2(latents[name], latents["dense"]),
            "replaced_blocks_first_sample": int(wrappers[name].replaced_blocks),
        }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
