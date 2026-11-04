#!/usr/bin/env python3
"""Training-free RDD cross-stage sparse-attention sampling oracle.

This script intentionally computes a dense QK matrix and applies structured
masks in PyTorch.  It validates generation quality before a block-sparse kernel
is written; it is not a speed benchmark.

For the Unified model, the final 16x16-stage attention route and K/V tensors are
cached and reused by the 32x32 stage.  For the Multistage B/L/XL model, only the
head-averaged spatial route is transferred because independently trained L and
XL K/V representations are not directly compatible.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import types
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

REPO_ROOT = Path(__file__).resolve().parents[2]
DIT_ROOT = REPO_ROOT / "DiT"
sys.path.insert(0, str(DIT_ROOT))
from models import DiT_models  # noqa: E402
from models_multires import DiTMultiRes_models, apply_rotary_emb  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


RESOLUTIONS_FINE_TO_COARSE = (32, 16, 8)
STAGE_LENGTHS = (300, 300, 400)
BOUNDARIES = (0, 300, 600, 1000)
SAMPLE_STEPS_COARSE_TO_FINE = (80, 80, 90)
DEFAULT_CLASSES = (207, 282, 323, 817)
CLASS_NAMES = {
    207: "golden retriever",
    282: "tiger cat",
    323: "monarch butterfly",
    817: "sports car",
}


@dataclass(frozen=True)
class SparseSetting:
    name: str
    fine_call_limit: int | None
    mode: str
    density: float = 0.4
    add_self_root: bool = True

    def layer_policy(self, layer: int, depth: int) -> tuple[float, bool]:
        if self.mode == "route":
            return self.density, False
        if self.mode != "hybrid":
            raise ValueError(f"unknown mode: {self.mode}")
        fraction = layer / max(depth - 1, 1)
        if fraction <= 0.60:
            return 0.20, True
        if fraction <= 0.85:
            return 0.40, False
        return 0.20, False


UNIFIED_SETTINGS = (
    SparseSetting("dense", 0, "route", 1.0),
    SparseSetting("route40_early30", 30, "route", 0.40),
    SparseSetting("route40_all90", None, "route", 0.40),
    SparseSetting("hybrid_early30", 30, "hybrid"),
    SparseSetting("hybrid_all90", None, "hybrid"),
)

MULTISTAGE_SETTINGS = (
    SparseSetting("dense", 0, "route", 1.0),
    SparseSetting("route40_early30", 30, "route", 0.40),
    SparseSetting("route40_all90", None, "route", 0.40),
)


def infer_num_classes(weights: dict[str, torch.Tensor]) -> int:
    embedding = weights.get("y_embedder.embedding_table.weight")
    return 0 if embedding is None else int(embedding.shape[0] - 1)


def load_weights(path: Path) -> dict[str, torch.Tensor]:
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("ema", checkpoint.get("model", checkpoint))


def load_unified(path: Path, device: torch.device) -> tuple[torch.nn.Module, int]:
    weights = load_weights(path)
    num_classes = infer_num_classes(weights)
    model = DiTMultiRes_models["DiT-XL/2"](num_classes=num_classes, base_size=16)
    model.load_state_dict(weights)
    model.to(device).eval().requires_grad_(False)
    del weights
    gc.collect()
    return model, num_classes


def load_multistage(
    directory: Path, device: torch.device
) -> tuple[dict[int, torch.nn.Module], int]:
    architectures = {8: "DiT-B/2", 16: "DiT-L/2", 32: "DiT-XL/2"}
    models: dict[int, torch.nn.Module] = {}
    num_classes = None
    for resolution in (8, 16, 32):
        weights = load_weights(
            directory / f"checkpoints_{resolution}x{resolution}" / "0400000.pt"
        )
        classes = infer_num_classes(weights)
        num_classes = classes if num_classes is None else num_classes
        if classes != num_classes:
            raise ValueError("class-count mismatch across Multistage checkpoints")
        model = DiT_models[architectures[resolution]](
            input_size=resolution,
            num_classes=classes,
            d_prediction="d_final_layer.linear.weight" in weights,
        )
        model.load_state_dict(weights)
        model.to(device).eval().requires_grad_(False)
        models[resolution] = model
        del weights
        gc.collect()
    assert num_classes is not None
    return models, num_classes


def create_sampling_diffusions(sample_steps: tuple[int, int, int]) -> dict[int, Any]:
    result = {}
    steps_by_resolution = dict(zip((8, 16, 32), sample_steps))
    for index, resolution in enumerate(RESOLUTIONS_FINE_TO_COARSE):
        result[resolution] = create_diffusion_with_drift(
            str(steps_by_resolution[resolution]),
            c=index if index < 2 else None,
            T_c=BOUNDARIES[index],
            T_c_plus_1=BOUNDARIES[index + 1],
        )
    return result


def upsample_with_noise(
    pred_xstart: torch.Tensor,
    diffusion: Any,
    target_size: int,
) -> torch.Tensor:
    scale = target_size // pred_xstart.shape[-1]
    upsampled = pred_xstart.repeat_interleave(scale, 2).repeat_interleave(scale, 3)
    local_t = torch.full(
        (upsampled.shape[0],),
        diffusion.num_timesteps - 1,
        device=upsampled.device,
        dtype=torch.long,
    )
    noise = torch.randn_like(upsampled)
    sqrt_alpha = _extract_into_tensor(
        diffusion.sqrt_alphas_cumprod, local_t, upsampled.shape
    )
    sqrt_one_minus = _extract_into_tensor(
        diffusion.sqrt_one_minus_alphas_cumprod, local_t, upsampled.shape
    )
    return sqrt_alpha * upsampled + sqrt_one_minus * noise


def root_mapping(grid: int, tile: int, device: torch.device) -> torch.Tensor:
    y, x = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )
    roots_axis = grid // tile
    return ((y // tile) * roots_axis + (x // tile)).flatten().long()


class CrossStageController:
    def __init__(
        self,
        setting: SparseSetting,
        unified: bool,
        source_depth: int,
        target_depth: int,
    ):
        self.setting = setting
        self.unified = unified
        self.source_depth = source_depth
        self.target_depth = target_depth
        self.resolution: int | None = None
        self.fine_call_index = -1
        self.source_cache: dict[int, dict[str, torch.Tensor]] = {}
        self.stats = {
            "fine_calls": 0,
            "sparse_fine_calls": 0,
            "layer_invocations": 0,
            "summary_invocations": 0,
            "effective_root_density_sum": 0.0,
        }

    def begin_forward(self, x: torch.Tensor) -> None:
        self.resolution = int(x.shape[-1])
        if self.resolution == 32:
            self.fine_call_index += 1
            self.stats["fine_calls"] += 1
            if self.is_sparse_call:
                self.stats["sparse_fine_calls"] += 1

    def end_forward(self) -> None:
        self.resolution = None

    @property
    def is_sparse_call(self) -> bool:
        if self.resolution != 32 or self.setting.name == "dense":
            return False
        limit = self.setting.fine_call_limit
        return limit is None or self.fine_call_index < limit

    def source_layer_for(self, target_layer: int) -> int:
        if self.unified:
            return target_layer
        fraction = target_layer / max(self.target_depth - 1, 1)
        return round(fraction * (self.source_depth - 1))

    def capture(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> None:
        attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
        self.source_cache[layer] = {
            "route": attention.detach(),
            "k": k.detach(),
            "v": v.detach(),
        }

    def sparse_attention(
        self,
        layer: int,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        source_layer = self.source_layer_for(layer)
        if source_layer not in self.source_cache:
            raise RuntimeError(
                f"missing 16-stage cache for target layer {layer} (source {source_layer})"
            )
        cache = self.source_cache[source_layer]
        route = cache["route"].to(device=q.device, dtype=torch.float32)
        if not self.unified:
            route = route.mean(1, keepdim=True).expand(-1, q.shape[1], -1, -1)
        elif route.shape[1] != q.shape[1]:
            raise RuntimeError("Unified head-count changed across resolutions")

        batch, heads, fine_tokens, _ = q.shape
        fine_grid = int(math.isqrt(fine_tokens))
        if fine_grid * fine_grid != fine_tokens or fine_grid % 2:
            raise ValueError(f"unexpected fine token grid: {fine_tokens}")
        query_root = root_mapping(fine_grid, 2, q.device)
        key_root = query_root
        roots = int(query_root.max()) + 1
        if route.shape[-2:] != (roots, roots):
            raise RuntimeError(
                f"cached route shape {tuple(route.shape)} incompatible with {roots} roots"
            )

        density, use_summary = self.setting.layer_policy(layer, self.target_depth)
        keep = max(1, math.ceil(density * roots))
        route_for_fine_query = route[:, :, query_root, :]
        selected = torch.zeros_like(route_for_fine_query, dtype=torch.bool)
        selected.scatter_(-1, route_for_fine_query.topk(keep, dim=-1).indices, True)
        if self.setting.add_self_root:
            selected.scatter_(-1, query_root.view(1, 1, -1, 1).expand(batch, heads, -1, 1), True)
        effective_density = float(selected.float().sum(-1).mean().item() / roots)
        self.stats["effective_root_density_sum"] += effective_density
        self.stats["layer_invocations"] += 1

        logits = q @ k.transpose(-2, -1) * scale
        exact_mask = selected[:, :, :, key_root]
        exact_logits = logits.masked_fill(~exact_mask, -torch.inf)

        if use_summary and self.unified:
            source_k = cache["k"].to(device=q.device, dtype=q.dtype)
            source_v = cache["v"].to(device=q.device, dtype=v.dtype)
            if source_k.shape[:2] != q.shape[:2] or source_k.shape[-1] != q.shape[-1]:
                raise RuntimeError("source K/V cannot be used as Unified summary")
            summary_logits = q @ source_k.transpose(-2, -1) * scale
            # Each 16-stage token represents a 2x2 group of four 32-stage tokens.
            summary_logits = summary_logits + math.log(4.0)
            summary_logits = summary_logits.masked_fill(selected, -torch.inf)
            weights = torch.cat((exact_logits, summary_logits), dim=-1).softmax(-1)
            output = weights[..., :fine_tokens] @ v
            output = output + weights[..., fine_tokens:] @ source_v
            self.stats["summary_invocations"] += 1
            return output

        return exact_logits.softmax(-1) @ v

    def report(self) -> dict[str, Any]:
        invocations = max(int(self.stats["layer_invocations"]), 1)
        return {
            **self.stats,
            "mean_effective_root_density": (
                float(self.stats["effective_root_density_sum"]) / invocations
            ),
            "setting": self.setting.name,
        }


def _qkv(
    module: torch.nn.Module,
    args: tuple[torch.Tensor, ...],
    unified: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    x = args[0]
    batch, seq, width = x.shape
    heads = module.num_heads
    head_dim = width // heads
    qkv = module.qkv(x).reshape(batch, seq, 3, heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if unified:
        q = apply_rotary_emb(q, args[1], args[2])
        k = apply_rotary_emb(k, args[1], args[2])
    else:
        q = module.q_norm(q)
        k = module.k_norm(k)
    return q, k, v, float(getattr(module, "scale", head_dim**-0.5))


@contextmanager
def patched_attention(
    models: dict[int, torch.nn.Module],
    controller: CrossStageController,
    unified: bool,
):
    records: list[tuple[torch.nn.Module, Any]] = []
    unique_models = list(dict.fromkeys(models.values()))
    for model in unique_models:
        for layer, block in enumerate(model.blocks):
            module = block.attn
            original = module.forward

            def replacement(self, *args, _layer=layer, _original=original):
                if controller.resolution not in (16, 32):
                    return _original(*args)
                q, k, v, scale = _qkv(self, args, unified)
                if controller.resolution == 16:
                    controller.capture(_layer, q, k, v, scale)
                    return _original(*args)
                if not controller.is_sparse_call:
                    return _original(*args)
                output = controller.sparse_attention(_layer, q, k, v, scale)
                x = output.transpose(1, 2).reshape(args[0].shape)
                x = self.proj(x)
                if hasattr(self, "proj_drop"):
                    x = self.proj_drop(x)
                return x

            records.append((module, original))
            module.forward = types.MethodType(replacement, module)
    try:
        yield
    finally:
        for module, original in records:
            module.forward = original


def cfg_model_call(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    y: torch.Tensor,
    cfg_scale: float,
) -> torch.Tensor:
    half = x[: len(x) // 2]
    combined = torch.cat((half, half), 0)
    output = model(combined, t, y)
    eps, rest = output[:, :3], output[:, 3:]
    cond_eps, uncond_eps = torch.chunk(eps, 2, dim=0)
    guided = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
    eps = torch.cat((guided, guided), 0)
    if getattr(model, "d_prediction", False):
        cond_rest, _ = torch.chunk(rest, 2, dim=0)
        rest = torch.cat((cond_rest, cond_rest), 0)
    return torch.cat((eps, rest), 1)


@torch.inference_mode()
def sample_variant(
    models: dict[int, torch.nn.Module],
    diffusions: dict[int, Any],
    vae: AutoencoderKL,
    labels: torch.Tensor,
    num_classes: int,
    cfg_scale: float,
    seed: int,
    controller: CrossStageController | None,
    unified: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch = labels.shape[0]
    use_cfg = cfg_scale > 1.0
    if use_cfg:
        y = torch.cat(
            (labels, torch.full_like(labels, num_classes)),
            dim=0,
        )
        batch_multiplier = 2
    else:
        y = labels
        batch_multiplier = 1

    pred_xstart = None
    final_sample = None
    elapsed = 0.0
    patch_context = (
        patched_attention(models, controller, unified)
        if controller is not None
        else nullcontext()
    )
    with patch_context:
        for stage_index, resolution in enumerate((8, 16, 32)):
            diffusion = diffusions[resolution]
            model = models[resolution]
            if stage_index == 0:
                state = torch.randn(
                    batch_multiplier * batch,
                    4,
                    resolution,
                    resolution,
                    device=labels.device,
                )
            else:
                assert pred_xstart is not None
                state = upsample_with_noise(pred_xstart, diffusion, resolution)
                if use_cfg:
                    state = torch.cat((state, state), 0)

            def model_forward(x, t, **kwargs):
                if controller is not None:
                    controller.begin_forward(x)
                try:
                    if use_cfg:
                        return cfg_model_call(model, x, t, kwargs["y"], cfg_scale)
                    return model(x, t, kwargs["y"])
                finally:
                    if controller is not None:
                        controller.end_forward()

            if labels.device.type == "cuda":
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
            if labels.device.type == "cuda":
                torch.cuda.synchronize()
            elapsed += time.perf_counter() - start
            final_sample = output["sample"].float()
            pred_xstart = output["pred_xstart"].float()
            if use_cfg:
                final_sample = final_sample[:batch]
                pred_xstart = pred_xstart[:batch]

    assert final_sample is not None
    decoded = vae.decode(final_sample / 0.18215).sample.float()
    report = {"denoiser_seconds": elapsed}
    if controller is not None:
        report.update(controller.report())
    return decoded, final_sample, report


def tensor_to_images(decoded: torch.Tensor) -> list[Image.Image]:
    array = (
        ((decoded.clamp(-1, 1) + 1) * 127.5)
        .byte()
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
    )
    return [Image.fromarray(image) for image in array]


def image_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().clamp(-1, 1)
    candidate = candidate.float().clamp(-1, 1)
    mse = float((reference - candidate).square().mean())
    mae = float((reference - candidate).abs().mean())
    psnr = float(10.0 * math.log10(4.0 / max(mse, 1e-12)))
    cosine = float(
        F.cosine_similarity(reference.flatten(1), candidate.flatten(1), dim=-1).mean()
    )
    return {"mse": mse, "mae": mae, "psnr_db": psnr, "pixel_cosine": cosine}


def save_comparison_grid(
    variants: dict[str, list[Image.Image]],
    labels: list[int],
    output: Path,
    title: str,
) -> None:
    names = list(variants)
    cell = next(iter(variants.values()))[0].width
    top = 58
    left = 170
    canvas = Image.new("RGB", (left + cell * len(names), top + cell * len(labels)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 5), title, fill="black", font=font)
    draw.text((8, 24), "same classes, initial noise, and transition noise", fill="black", font=font)
    for column, name in enumerate(names):
        draw.text((left + column * cell + 5, 42), name, fill="black", font=font)
    for row, label in enumerate(labels):
        name = CLASS_NAMES.get(label, str(label))
        draw.text((8, top + row * cell + 8), f"class {label}", fill="black", font=font)
        draw.text((8, top + row * cell + 25), name, fill="black", font=font)
        for column, variant_name in enumerate(names):
            canvas.paste(
                variants[variant_name][row],
                (left + column * cell, top + row * cell),
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_model_family(
    family: str,
    models: dict[int, torch.nn.Module],
    diffusions: dict[int, Any],
    vae: AutoencoderKL,
    labels: torch.Tensor,
    num_classes: int,
    cfg_scale: float,
    seed: int,
    output_dir: Path,
    settings: tuple[SparseSetting, ...],
    unified: bool,
) -> dict[str, Any]:
    images_by_variant: dict[str, list[Image.Image]] = {}
    decoded_by_variant: dict[str, torch.Tensor] = {}
    latent_by_variant: dict[str, torch.Tensor] = {}
    report: dict[str, Any] = {"family": family, "variants": {}}
    source_depth = len(models[16].blocks)
    target_depth = len(models[32].blocks)

    for setting in settings:
        print(f"[{family}] sampling {setting.name}", flush=True)
        controller = None
        if setting.name != "dense":
            controller = CrossStageController(
                setting=setting,
                unified=unified,
                source_depth=source_depth,
                target_depth=target_depth,
            )
        decoded, latent, setting_report = sample_variant(
            models,
            diffusions,
            vae,
            labels,
            num_classes,
            cfg_scale,
            seed,
            controller,
            unified,
        )
        decoded_cpu, latent_cpu = decoded.cpu(), latent.cpu()
        decoded_by_variant[setting.name] = decoded_cpu
        latent_by_variant[setting.name] = latent_cpu
        images = tensor_to_images(decoded_cpu)
        images_by_variant[setting.name] = images
        variant_dir = output_dir / family / setting.name
        variant_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(images):
            image.save(variant_dir / f"class_{int(labels[index])}_seed_{seed}.png")
        report["variants"][setting.name] = setting_report
        del decoded, latent
        gc.collect()
        torch.cuda.empty_cache()

    dense_decoded = decoded_by_variant["dense"]
    dense_latent = latent_by_variant["dense"]
    for name in images_by_variant:
        if name == "dense":
            continue
        report["variants"][name]["image_vs_dense"] = image_metrics(
            dense_decoded, decoded_by_variant[name]
        )
        report["variants"][name]["latent_vs_dense"] = image_metrics(
            dense_latent, latent_by_variant[name]
        )

    grid_path = output_dir / f"{family}_comparison.png"
    save_comparison_grid(
        images_by_variant,
        labels.cpu().tolist(),
        grid_path,
        title=f"RDD {family}: dense vs training-free cross-stage sparse attention",
    )
    report["comparison_grid"] = str(grid_path)
    print(f"[{family}] saved {grid_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt", required=True)
    parser.add_argument("--multistage-dir", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--families", default="unified,multistage")
    parser.add_argument("--classes", default=",".join(map(str, DEFAULT_CLASSES)))
    parser.add_argument("--cfg-scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--sample-steps",
        default=",".join(map(str, SAMPLE_STEPS_COARSE_TO_FINE)),
        help="coarse-to-fine sampling steps; use a tiny value only for smoke tests",
    )
    parser.add_argument(
        "--settings",
        default="all",
        help="comma-separated variant names, or all; dense is added automatically",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    torch.set_grad_enabled(False)
    labels = torch.tensor([int(x) for x in args.classes.split(",")], device=device)
    families = [x.strip() for x in args.families.split(",") if x.strip()]
    sample_steps = tuple(int(x) for x in args.sample_steps.split(","))
    if len(sample_steps) != 3:
        raise ValueError("--sample-steps must contain coarse,middle,fine values")
    requested_settings = None
    if args.settings != "all":
        requested_settings = {x.strip() for x in args.settings.split(",") if x.strip()}
        requested_settings.add("dense")

    def select_settings(settings: tuple[SparseSetting, ...]) -> tuple[SparseSetting, ...]:
        if requested_settings is None:
            return settings
        selected = tuple(item for item in settings if item.name in requested_settings)
        unknown = requested_settings - {item.name for item in settings}
        if unknown:
            raise ValueError(f"settings unavailable for selected family: {sorted(unknown)}")
        return selected

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("loading local SD VAE", flush=True)
    vae = AutoencoderKL.from_pretrained(
        args.vae_path, local_files_only=True
    ).to(device).eval().requires_grad_(False)
    diffusions = create_sampling_diffusions(sample_steps)
    final_report: dict[str, Any] = {
        "scope": "training-free correctness oracle; dense QK plus structured mask",
        "classes": labels.cpu().tolist(),
        "cfg_scale": args.cfg_scale,
        "seed": args.seed,
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "sample_steps_coarse_to_fine": list(sample_steps),
        "upsample": "nearest-neighbor noise filling",
        "families": {},
    }

    if "unified" in families:
        model, classes = load_unified(Path(args.unified_ckpt), device)
        models = {8: model, 16: model, 32: model}
        final_report["families"]["unified_xl"] = run_model_family(
            "unified_xl",
            models,
            diffusions,
            vae,
            labels,
            classes,
            args.cfg_scale,
            args.seed,
            output_dir,
            select_settings(UNIFIED_SETTINGS),
            unified=True,
        )
        del model, models
        gc.collect()
        torch.cuda.empty_cache()

    if "multistage" in families:
        models, classes = load_multistage(Path(args.multistage_dir), device)
        final_report["families"]["multistage_blxl"] = run_model_family(
            "multistage_blxl",
            models,
            diffusions,
            vae,
            labels,
            classes,
            args.cfg_scale,
            args.seed,
            output_dir,
            select_settings(MULTISTAGE_SETTINGS),
            unified=False,
        )
        del models
        gc.collect()
        torch.cuda.empty_cache()

    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(final_report, indent=2), encoding="utf-8")
    print(f"saved {report_path}", flush=True)


if __name__ == "__main__":
    main()
