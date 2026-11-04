#!/usr/bin/env python3
"""Training-free sharing of self-attention probability across CFG branches.

The RDD ImageNet sampler evaluates conditional and unconditional branches in a
single 2B batch.  This experiment computes Q/K only for the conditional half,
uses its probability matrix for both halves, and always computes each branch's
current V.  An optional temporal interval also reuses that conditional
probability for a few denoising calls.

This is deliberately isolated from the existing sampler so old checkpoints and
scripts retain bit-for-bit behavior.
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from diffusers.models import AutoencoderKL
from PIL import Image

from sample_rdd_cross_stage_sparse import (
    DEFAULT_CLASSES,
    STAGE_LENGTHS,
    _qkv,
    cfg_model_call,
    create_sampling_diffusions,
    image_metrics,
    load_unified,
    save_comparison_grid,
    tensor_to_images,
    upsample_with_noise,
)
from sample_rdd_hierarchical_kernel import v_only


@dataclass(frozen=True)
class CFGShareSetting:
    name: str
    stages: tuple[int, ...]
    interval: int = 1
    layers: int = 28


SETTINGS = (
    CFGShareSetting("cfg_share_16_32_k1", (16, 32), 1, 28),
    CFGShareSetting("cfg_share_all_k1", (8, 16, 32), 1, 28),
    CFGShareSetting("cfg_share_all_k2", (8, 16, 32), 2, 28),
    CFGShareSetting("cfg_share_all_k3", (8, 16, 32), 3, 28),
    CFGShareSetting("cfg_share_16_32_k3", (16, 32), 3, 28),
    CFGShareSetting("cfg_share_all_k1_first7", (8, 16, 32), 1, 7),
)


class CFGShareController:
    def __init__(self, setting: CFGShareSetting, total_calls: dict[int, int]):
        self.setting = setting
        self.total_calls = total_calls
        self.resolution: int | None = None
        self.call_index = {8: -1, 16: -1, 32: -1}
        self.probabilities: dict[int, dict[int, torch.Tensor]] = {
            8: {},
            16: {},
            32: {},
        }
        self.stats = {
            "cfg_probability_refresh_calls": 0,
            "cfg_probability_temporal_hits": 0,
            "unconditional_qk_skipped_calls": 0,
            "full_attention_fallback_calls": 0,
        }

    def begin_forward(self, x: torch.Tensor) -> None:
        self.resolution = int(x.shape[-1])
        self.call_index[self.resolution] += 1

    def end_forward(self) -> None:
        self.resolution = None

    def enabled(self, layer: int, batch: int) -> bool:
        return (
            self.resolution in self.setting.stages
            and layer < self.setting.layers
            and batch % 2 == 0
        )

    def is_last_call(self) -> bool:
        assert self.resolution is not None
        return self.call_index[self.resolution] == self.total_calls[self.resolution] - 1

    def cache_hit(self, layer: int) -> bool:
        assert self.resolution is not None
        return (
            self.setting.interval > 1
            and layer in self.probabilities[self.resolution]
            and self.call_index[self.resolution] % self.setting.interval != 0
            and not self.is_last_call()
        )

    def store(self, layer: int, probability: torch.Tensor) -> None:
        assert self.resolution is not None
        self.probabilities[self.resolution][layer] = probability.detach()

    def cached(self, layer: int) -> torch.Tensor:
        assert self.resolution is not None
        return self.probabilities[self.resolution][layer]

    def report(self) -> dict[str, Any]:
        return {"setting": asdict(self.setting), **self.stats}


@contextmanager
def patched_cfg_share_attention(model: torch.nn.Module, controller: CFGShareController):
    records = []
    for layer, block in enumerate(model.blocks):
        module = block.attn
        original = module.forward

        def replacement(self, *args, _layer=layer, _original=original):
            x = args[0]
            if not controller.enabled(_layer, x.shape[0]):
                controller.stats["full_attention_fallback_calls"] += 1
                return _original(*args)

            half = x.shape[0] // 2
            if controller.cache_hit(_layer):
                value = v_only(self, x)
                probability = controller.cached(_layer).to(value.dtype)
                output = torch.cat(
                    (probability @ value[:half], probability @ value[half:]), dim=0
                )
                controller.stats["cfg_probability_temporal_hits"] += 1
            else:
                conditional_args = (x[:half], *args[1:])
                q, k, v_conditional, scale = _qkv(self, conditional_args, unified=True)
                v_unconditional = v_only(self, x[half:])
                probability = (q @ k.transpose(-2, -1) * scale).softmax(-1)
                controller.store(_layer, probability)
                output = torch.cat(
                    (probability @ v_conditional, probability @ v_unconditional), dim=0
                )
                controller.stats["cfg_probability_refresh_calls"] += 1

            controller.stats["unconditional_qk_skipped_calls"] += 1
            output = output.transpose(1, 2).reshape(x.shape)
            return self.proj(output)

        records.append((module, original))
        module.forward = types.MethodType(replacement, module)
    try:
        yield
    finally:
        for module, original in records:
            module.forward = original


@torch.inference_mode()
def sample_variant(
    model,
    diffusions,
    vae,
    labels,
    num_classes,
    cfg_scale,
    seed,
    controller: CFGShareController | None,
):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cfg_scale <= 1.0:
        raise ValueError("CFG sharing requires --cfg-scale > 1")
    y = torch.cat((labels, torch.full_like(labels, num_classes)))
    pred_xstart = None
    final_sample = None
    elapsed = 0.0
    stage_seconds = {}
    context = patched_cfg_share_attention(model, controller) if controller else nullcontext()
    with context:
        for stage, resolution in enumerate((8, 16, 32)):
            diffusion = diffusions[resolution]
            if stage == 0:
                state = torch.randn(2 * len(labels), 4, resolution, resolution, device=labels.device)
            else:
                state = upsample_with_noise(pred_xstart, diffusion, resolution)
                state = torch.cat((state, state))

            def model_forward(x, t, **kwargs):
                if controller:
                    controller.begin_forward(x)
                try:
                    return cfg_model_call(model, x, t, kwargs["y"], cfg_scale)
                finally:
                    if controller:
                        controller.end_forward()

            torch.cuda.synchronize()
            start = time.perf_counter()
            output = diffusion.p_sample_loop(
                model_forward,
                state.shape,
                noise=state,
                clip_denoised=False,
                model_kwargs={"y": y},
                progress=False,
                device=labels.device,
            )
            torch.cuda.synchronize()
            stage_seconds[str(resolution)] = time.perf_counter() - start
            elapsed += stage_seconds[str(resolution)]
            final_sample = output["sample"].float()[: len(labels)]
            pred_xstart = output["pred_xstart"].float()[: len(labels)]

    decoded = vae.decode(final_sample / 0.18215).sample.float()
    stats = {"denoiser_seconds": elapsed, "stage_seconds": stage_seconds}
    if controller:
        stats.update(controller.report())
    return decoded, final_sample, stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--classes", default=",".join(map(str, DEFAULT_CLASSES)))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--sample-steps", default="80,80,90")
    parser.add_argument("--settings", default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.45)
    parser.add_argument("--timing-repeats", type=int, default=1)
    args = parser.parse_args()

    device = torch.device(args.device)
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    steps = tuple(int(x) for x in args.sample_steps.split(","))
    total_calls = dict(zip((8, 16, 32), steps))
    requested = None if args.settings == "all" else set(args.settings.split(","))
    settings = tuple(s for s in SETTINGS if requested is None or s.name in requested)
    unknown = set() if requested is None else requested - {s.name for s in SETTINGS}
    if unknown:
        raise ValueError(f"unknown settings: {sorted(unknown)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    model, num_classes = load_unified(Path(args.unified_ckpt), device)
    diffusions = create_sampling_diffusions(steps)

    images: dict[str, list[Image.Image]] = {}
    decoded: dict[str, torch.Tensor] = {}
    latents: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {
        "scope": "training-free conditional-to-unconditional attention probability sharing",
        "model": "RDD Unified XL/2 400K",
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(steps),
        "variants": {},
    }

    # Untimed dense warmup, then repeated same-noise baseline.
    sample_variant(model, diffusions, vae, labels[:1], num_classes, args.cfg_scale, args.seed, None)
    dense_runs = []
    dense_stage_runs = []
    for _ in range(args.timing_repeats):
        out, latent, stats = sample_variant(
            model, diffusions, vae, labels, num_classes, args.cfg_scale, args.seed, None
        )
        dense_runs.append(stats["denoiser_seconds"])
        dense_stage_runs.append(stats["stage_seconds"])
    stats["denoiser_seconds"] = statistics.median(dense_runs)
    stats["denoiser_seconds_runs"] = dense_runs
    stats["stage_seconds"] = {
        r: statistics.median(run[r] for run in dense_stage_runs) for r in ("8", "16", "32")
    }
    decoded["dense"], latents["dense"] = out.cpu(), latent.cpu()
    images["dense"] = tensor_to_images(decoded["dense"])
    report["variants"]["dense"] = stats

    for setting in settings:
        print(f"sampling {setting.name}", flush=True)
        runs = []
        stage_runs = []
        for _ in range(args.timing_repeats):
            controller = CFGShareController(setting, total_calls)
            out, latent, stats = sample_variant(
                model,
                diffusions,
                vae,
                labels,
                num_classes,
                args.cfg_scale,
                args.seed,
                controller,
            )
            runs.append(stats["denoiser_seconds"])
            stage_runs.append(stats["stage_seconds"])
        stats["denoiser_seconds"] = statistics.median(runs)
        stats["denoiser_seconds_runs"] = runs
        stats["stage_seconds"] = {
            r: statistics.median(run[r] for run in stage_runs) for r in ("8", "16", "32")
        }
        decoded[setting.name], latents[setting.name] = out.cpu(), latent.cpu()
        images[setting.name] = tensor_to_images(decoded[setting.name])
        stats["image_vs_dense"] = image_metrics(decoded["dense"], decoded[setting.name])
        stats["latent_vs_dense"] = image_metrics(latents["dense"], latents[setting.name])
        report["variants"][setting.name] = stats

        variant_dir = output_dir / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images[setting.name]):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")
        gc.collect()
        torch.cuda.empty_cache()

    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    for index, image in enumerate(images["dense"]):
        image.save(dense_dir / f"class_{int(labels[index])}_seed_{args.seed}.png")

    grid = output_dir / "cfg_probability_share_comparison.png"
    save_comparison_grid(
        images,
        labels.cpu().tolist(),
        grid,
        "RDD Unified XL: CFG attention-probability sharing",
    )
    report["comparison_grid"] = str(grid)
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"saved {grid}", flush=True)


if __name__ == "__main__":
    main()
