#!/usr/bin/env python3
"""Fair hidden/block-residual reuse probe: trained RDD vs original DiT.

RDD caches block updates at the 16x16->32x32 boundary (global t=300 to
t=299). DiT caches full-resolution updates at t=300 and reuses them at t=299.
Both arms use XL/2 400K checkpoints and identical images, labels, noise,
timesteps, and skipped-block budgets. This is a diagnostic, not a FID test.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
DIT_ROOT = REPO_ROOT / "DiT"
sys.path.insert(0, str(DIT_ROOT))
from models import DiT_models  # noqa: E402
from models_multires import DiTMultiRes_models  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion, create_diffusion_with_drift  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


def center_crop_arr(image: Image.Image, size: int) -> Image.Image:
    while min(image.size) >= 2 * size:
        image = image.resize(tuple(x // 2 for x in image.size), Image.Resampling.BOX)
    scale = size / min(image.size)
    image = image.resize(tuple(round(x * scale) for x in image.size), Image.Resampling.BICUBIC)
    array = np.asarray(image)
    top = (array.shape[0] - size) // 2
    left = (array.shape[1] - size) // 2
    return Image.fromarray(array[top : top + size, left : left + size])


def discover_train_images(root: Path, count: int, offset: int) -> tuple[list[Path], torch.Tensor]:
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    if len(classes) < count:
        raise RuntimeError(f"expected ImageNet class folders under {root}")
    paths: list[Path] = []
    labels: list[int] = []
    for step in range(count):
        label = (offset + step * 97) % len(classes)
        files = sorted(
            path for path in classes[label].iterdir()
            if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not files:
            raise RuntimeError(f"no image in {classes[label]}")
        paths.append(files[(offset + step * 13) % len(files)])
        labels.append(label)
    return paths, torch.tensor(labels, dtype=torch.long)


@torch.inference_mode()
def encode_images(paths: list[Path], vae_path: Path, device: torch.device) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    pixels = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
    vae = AutoencoderKL.from_pretrained(str(vae_path), local_files_only=True)
    vae.to(device).eval().requires_grad_(False)
    latent = vae.encode(pixels.to(device)).latent_dist.mode() * 0.18215
    del vae, pixels
    gc.collect()
    torch.cuda.empty_cache()
    return latent.float()


def checkpoint_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("ema", checkpoint.get("model", checkpoint))


def infer_num_classes(weights: dict[str, torch.Tensor]) -> int:
    return int(weights["y_embedder.embedding_table.weight"].shape[0] - 1)


def load_rdd(path: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    weights = checkpoint_weights(path)
    model = DiTMultiRes_models["DiT-XL/2"](num_classes=infer_num_classes(weights), base_size=16)
    model.load_state_dict(weights)
    del weights
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def load_dit(path: Path, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    weights = checkpoint_weights(path)
    model = DiT_models["DiT-XL/2"](input_size=32, num_classes=infer_num_classes(weights))
    model.load_state_dict(weights)
    del weights
    return model.to(device=device, dtype=dtype).eval().requires_grad_(False)


def aligned_noise(latent32: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=latent32.device).manual_seed(seed)
    eps32 = torch.randn(latent32.shape, device=latent32.device, generator=generator)
    return eps32, F.avg_pool2d(eps32, 2, 2) * 2.0


def upsample_tokens(tokens: torch.Tensor, target_grid: int, mode: str = "nearest") -> torch.Tensor:
    batch, count, channels = tokens.shape
    grid = int(round(math.sqrt(count)))
    if grid * grid != count:
        raise ValueError(f"token count {count} is not square")
    spatial = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
    if mode == "nearest":
        spatial = F.interpolate(spatial, size=(target_grid, target_grid), mode="nearest")
    else:
        spatial = F.interpolate(spatial, size=(target_grid, target_grid), mode=mode, align_corners=False)
    return spatial.flatten(2).transpose(1, 2)


def low_high_tokens(tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch, count, channels = tokens.shape
    grid = int(round(math.sqrt(count)))
    spatial = tokens.transpose(1, 2).reshape(batch, channels, grid, grid)
    low = F.interpolate(F.avg_pool2d(spatial, 2, 2), size=(grid, grid), mode="nearest")
    low = low.flatten(2).transpose(1, 2)
    return low, tokens - low


def tensor_metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    candidate = candidate.float().flatten(1)
    reference = reference.float().flatten(1)
    error = candidate - reference
    return {
        "cosine": float(F.cosine_similarity(candidate, reference, dim=1).mean()),
        "relative_l2": float((error.square().sum(1) / reference.square().sum(1).clamp_min(1e-12)).sqrt().mean()),
        "rmse": float(error.square().mean().sqrt()),
    }


@torch.inference_mode()
def _manual_forward_impl(
    model: torch.nn.Module,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    labels: torch.Tensor,
    *,
    multires: bool,
    replacements: dict[int, torch.Tensor] | None = None,
    replacement_mode: str = "residual",
    capture: bool = False,
    capture_layers: set[int] | None = None,
    conditioned_replacements: bool = False,
) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    _, _, height, width = latent.shape
    if multires:
        grid_h, grid_w = height // model.patch_size, width // model.patch_size
        hidden = model.x_embedder(latent)
        rope_cos, rope_sin = model.get_rope(grid_h, grid_w, latent.device)
    else:
        grid_h = grid_w = height // model.patch_size
        hidden = model.x_embedder(latent) + model.pos_embed
        rope_cos = rope_sin = None
    condition = model.t_embedder(timestep) + model.y_embedder(labels, False)
    traces: dict[int, dict[str, torch.Tensor]] = {}
    replacements = replacements or {}
    for layer, block in enumerate(model.blocks):
        before = hidden
        if layer in replacements:
            replacement_spec = replacements[layer]
            if callable(replacement_spec):
                replacement = (replacement_spec(before, condition) if conditioned_replacements
                               else replacement_spec(before))
            else:
                replacement = replacement_spec
            replacement = replacement.to(device=hidden.device, dtype=hidden.dtype)
            if replacement_mode == "residual":
                hidden = before + replacement
            elif replacement_mode == "hidden":
                hidden = replacement
            else:
                raise ValueError(replacement_mode)
        elif multires:
            hidden = block(hidden, condition, rope_cos, rope_sin)
        else:
            hidden = block(hidden, condition)
        if capture or (capture_layers is not None and layer in capture_layers):
            traces[layer] = {
                "input": before.detach(),
                "output": hidden.detach(),
                "residual": (hidden - before).detach(),
            }
    output = model.final_layer(hidden, condition)
    output = model.unpatchify(output, grid_h, grid_w) if multires else model.unpatchify(output)
    return output, traces


@torch.inference_mode()
def manual_forward(*args: Any, **kwargs: Any) -> tuple[torch.Tensor, dict[int, dict[str, torch.Tensor]]]:
    """Run the manual block path with the same AMP policy as training/sampling."""
    latent = args[1]
    with torch.autocast(
        device_type=latent.device.type,
        dtype=torch.bfloat16,
        enabled=latent.device.type == "cuda",
    ):
        return _manual_forward_impl(*args, **kwargs)


def pred_x0(diffusion: Any, output: torch.Tensor, x_t: torch.Tensor, local_t: torch.Tensor) -> torch.Tensor:
    def constant_model(_x: torch.Tensor, _t: torch.Tensor, **_kwargs: Any) -> torch.Tensor:
        return output
    return diffusion.p_mean_variance(
        constant_model, x_t, local_t, clip_denoised=False, model_kwargs={}
    )["pred_xstart"].float()


def stage_drift(diffusion: Any, clean: torch.Tensor) -> torch.Tensor | None:
    if diffusion.c is None:
        return None
    return diffusion.upsample(diffusion.downsample(clean, diffusion.sample_scale), diffusion.sample_scale) - clean


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def benchmark_forward(function: Any, device: torch.device, warmup: int, rounds: int) -> dict[str, float]:
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(rounds):
        start = time.perf_counter()
        function()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000.0)
    return {"mean_ms": float(np.mean(samples)), "median_ms": float(np.median(samples)), "std_ms": float(np.std(samples))}


def prepare_replacements(
    source: dict[int, dict[str, torch.Tensor]],
    layers: list[int],
    target_grid: int,
    mode: str,
    interpolation: str,
) -> dict[int, torch.Tensor]:
    key = "residual" if mode == "residual" else "output"
    result = {}
    for layer in layers:
        value = source[layer][key]
        if value.shape[1] != target_grid * target_grid:
            value = upsample_tokens(value, target_grid, interpolation)
        result[layer] = value
    return result


@torch.inference_mode()
def evaluate_arm(
    name: str,
    model: torch.nn.Module,
    multires: bool,
    source_x: torch.Tensor,
    source_t_model: torch.Tensor,
    target_x: torch.Tensor,
    target_t_model: torch.Tensor,
    target_t_local: torch.Tensor,
    labels: torch.Tensor,
    diffusion: Any,
    interpolation: str,
    warmup: int,
    rounds: int,
) -> dict[str, Any]:
    dtype = model.x_embedder.proj.weight.dtype
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        _, source_trace = manual_forward(model, source_x.to(dtype), source_t_model, labels, multires=multires, capture=True)
        exact_output, target_trace = manual_forward(model, target_x.to(dtype), target_t_model, labels, multires=multires, capture=True)
    exact_x0 = pred_x0(diffusion, exact_output, target_x, target_t_local)
    depth = len(model.blocks)
    target_grid = int(round(math.sqrt(target_trace[0]["residual"].shape[1])))
    hidden_layers = {0, depth // 4, depth // 2, 3 * depth // 4, depth - 1}
    single_layers = []
    for layer in range(depth):
        residual = prepare_replacements(source_trace, [layer], target_grid, "residual", interpolation)[layer]
        target_residual = target_trace[layer]["residual"]
        low, high = low_high_tokens(target_residual)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            replaced_output, _ = manual_forward(
                model, target_x.to(dtype), target_t_model, labels, multires=multires,
                replacements={layer: residual}, replacement_mode="residual",
            )
        replaced_x0 = pred_x0(diffusion, replaced_output, target_x, target_t_local)
        row: dict[str, Any] = {
            "layer": layer,
            "source_vs_target_residual": tensor_metrics(residual, target_residual),
            "source_vs_target_low_frequency": tensor_metrics(residual, low),
            "source_vs_target_high_frequency": tensor_metrics(residual, high),
            "model_output": tensor_metrics(replaced_output[:, :4], exact_output[:, :4]),
            "pred_x0": tensor_metrics(replaced_x0, exact_x0),
        }
        if layer in hidden_layers:
            hidden = prepare_replacements(source_trace, [layer], target_grid, "hidden", interpolation)[layer]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden_output, _ = manual_forward(
                    model, target_x.to(dtype), target_t_model, labels, multires=multires,
                    replacements={layer: hidden}, replacement_mode="hidden",
                )
            row["whole_hidden_replacement"] = {
                "model_output": tensor_metrics(hidden_output[:, :4], exact_output[:, :4]),
                "pred_x0": tensor_metrics(pred_x0(diffusion, hidden_output, target_x, target_t_local), exact_x0),
            }
        single_layers.append(row)

    layer_sets = {
        "tail4": list(range(depth - 4, depth)),
        "tail6": list(range(depth - 6, depth)),
        "tail8": list(range(depth - 8, depth)),
        "alternating6": [depth - 12, depth - 10, depth - 8, depth - 6, depth - 4, depth - 2],
        "uniform6": [round(x) for x in np.linspace(0, depth - 1, 6)],
    }
    combinations = {}
    for setting, layers in layer_sets.items():
        replacements = prepare_replacements(source_trace, layers, target_grid, "residual", interpolation)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            replaced_output, _ = manual_forward(
                model, target_x.to(dtype), target_t_model, labels, multires=multires,
                replacements=replacements, replacement_mode="residual",
            )
        combinations[setting] = {
            "layers": layers,
            "model_output": tensor_metrics(replaced_output[:, :4], exact_output[:, :4]),
            "pred_x0": tensor_metrics(pred_x0(diffusion, replaced_output, target_x, target_t_local), exact_x0),
        }

    baseline_fn = lambda: manual_forward(model, target_x.to(dtype), target_t_model, labels, multires=multires)
    latency = {"baseline": benchmark_forward(baseline_fn, target_x.device, warmup, rounds)}
    baseline_ms = latency["baseline"]["median_ms"]
    for setting in ("tail4", "tail6", "tail8", "alternating6"):
        replacements = prepare_replacements(source_trace, layer_sets[setting], target_grid, "residual", interpolation)
        function = lambda replacements=replacements: manual_forward(
            model, target_x.to(dtype), target_t_model, labels, multires=multires,
            replacements=replacements, replacement_mode="residual",
        )
        timing = benchmark_forward(function, target_x.device, warmup, rounds)
        timing["speedup_vs_baseline"] = baseline_ms / timing["median_ms"]
        timing["latency_reduction_percent"] = 100.0 * (1.0 - timing["median_ms"] / baseline_ms)
        latency[setting] = timing
    return {
        "arm": name,
        "depth": depth,
        "source_tokens": int(source_trace[0]["residual"].shape[1]),
        "target_tokens": int(target_trace[0]["residual"].shape[1]),
        "interpolation": interpolation,
        "single_layer": single_layers,
        "combinations": combinations,
        "latency": latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rdd-ckpt", required=True)
    parser.add_argument("--dit-ckpt", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-train", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--image-offset", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--interpolation", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--timing-rounds", type=int, default=15)
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    paths, labels_cpu = discover_train_images(Path(args.imagenet_train), args.batch_size, args.image_offset)
    latent32 = encode_images(paths, Path(args.vae_path), device)
    latent16 = F.avg_pool2d(latent32, 2, 2)
    labels = labels_cpu.to(device)
    eps32, eps16 = aligned_noise(latent32, args.seed)

    rdd32 = create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=0, T_c=0, T_c_plus_1=300)
    rdd16 = create_diffusion_with_drift(timestep_respacing="", diffusion_steps=1000, c=1, T_c=300, T_c_plus_1=600)
    source_t_local = torch.zeros(args.batch_size, device=device, dtype=torch.long)
    target_t_local = torch.full((args.batch_size,), 299, device=device, dtype=torch.long)
    source_rdd = rdd16.q_sample(latent16, source_t_local, d=stage_drift(rdd16, latent16), noise=eps16)
    target_rdd = rdd32.q_sample(latent32, target_t_local, d=stage_drift(rdd32, latent32), noise=eps32)
    source_t_global = torch.full((args.batch_size,), int(rdd16.timestep_map[0]), device=device, dtype=torch.long)
    target_t_global = torch.full((args.batch_size,), int(rdd32.timestep_map[299]), device=device, dtype=torch.long)

    rdd_model = load_rdd(Path(args.rdd_ckpt), device, torch.bfloat16)
    rdd_report = evaluate_arm(
        "rdd_cross_scale_16_t300_to_32_t299", rdd_model, True,
        source_rdd, source_t_global, target_rdd, target_t_global, target_t_local,
        labels, rdd32, args.interpolation, args.warmup, args.timing_rounds,
    )
    del rdd_model
    gc.collect()
    torch.cuda.empty_cache()

    dit_diffusion = create_diffusion(timestep_respacing="", diffusion_steps=1000)
    source_dit_t = torch.full((args.batch_size,), 300, device=device, dtype=torch.long)
    target_dit_t = torch.full((args.batch_size,), 299, device=device, dtype=torch.long)
    source_dit = dit_diffusion.q_sample(latent32, source_dit_t, noise=eps32)
    target_dit = dit_diffusion.q_sample(latent32, target_dit_t, noise=eps32)
    dit_model = load_dit(Path(args.dit_ckpt), device, torch.bfloat16)
    dit_report = evaluate_arm(
        "dit_full_resolution_t300_to_t299", dit_model, False,
        source_dit, source_dit_t, target_dit, target_dit_t, target_dit_t,
        labels, dit_diffusion, args.interpolation, args.warmup, args.timing_rounds,
    )

    report = {
        "schema": "rdd-vs-dit-block-reuse-v1",
        "seed": args.seed,
        "image_offset": args.image_offset,
        "images": [str(path) for path in paths],
        "labels": labels_cpu.tolist(),
        "fairness": {
            "architecture": "DiT-XL/2 for both arms",
            "checkpoint_step": 400000,
            "source_global_t": 300,
            "target_global_t": 299,
            "target_resolution": "32x32 latent / 256x256 image",
            "same_latents_labels_noise": True,
            "same_skipped_block_sets": True,
            "rdd_source": "16x16 stage boundary, spatially upsampled",
            "dit_source": "previous-timestep 32x32 full-resolution cache",
        },
        "rdd": rdd_report,
        "dit": dit_report,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
