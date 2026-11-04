#!/usr/bin/env python3
"""Probe cross-resolution structure in trained 2-D RDD checkpoints.

The experiment compares the last denoiser call of a lower-resolution stage
with the first denoiser call of the adjacent finer stage.  It measures:

* low-pass consistency of predicted x0 latents;
* attention-route similarity after both maps are aggregated to the same grid;
* fine-attention output error when routed by the previous stage;
* candidate-prior + fine residual routing;
* hidden/K/V similarity and parent-summary tail approximation.

The Unified model shares weights and hidden dimensions across resolutions.
The Multistage model intentionally uses B/L/XL models, so only topology is
directly reusable; hidden/K/V transfer necessarily requires an adapter.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
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
from models_multires import DiTMultiRes_models, apply_rotary_emb  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))
from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusers.models import AutoencoderKL  # noqa: E402


RESOLUTIONS = (32, 16, 8)
STAGE_LENGTHS = (300, 300, 400)
BOUNDARIES = (0, 300, 600, 1000)
SLOT_FRACTIONS = (0.0, 0.25, 0.5, 0.75, 1.0)


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.Resampling.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )
    arr = np.asarray(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def discover_images(root: Path, count: int) -> tuple[list[Path], torch.Tensor]:
    classes = sorted(path for path in root.iterdir() if path.is_dir())
    paths: list[Path] = []
    labels: list[int] = []
    # Round-robin over classes avoids accidentally probing a single class.
    class_files = [
        sorted(p for p in cls.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for cls in classes
    ]
    offset = 0
    while len(paths) < count:
        progressed = False
        for label, files in enumerate(class_files):
            if offset < len(files):
                paths.append(files[offset])
                labels.append(label)
                progressed = True
                if len(paths) == count:
                    break
        if not progressed:
            break
        offset += 1
    if len(paths) != count:
        raise RuntimeError(f"found only {len(paths)} images under {root}, need {count}")
    return paths, torch.tensor(labels, dtype=torch.long)


@torch.inference_mode()
def encode_images(
    paths: list[Path], vae_path: Path, device: torch.device
) -> torch.Tensor:
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda image: center_crop_arr(image, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ]
    )
    images = torch.stack([transform(Image.open(path).convert("RGB")) for path in paths])
    vae = AutoencoderKL.from_pretrained(str(vae_path), local_files_only=True).to(device)
    vae.eval().requires_grad_(False)
    # Deterministic posterior mode isolates cross-stage effects from VAE noise.
    latent = vae.encode(images.to(device)).latent_dist.mode() * 0.18215
    del vae, images
    gc.collect()
    torch.cuda.empty_cache()
    return latent.float()


def downsample_latent(x: torch.Tensor, target: int) -> torch.Tensor:
    factor = x.shape[-1] // target
    return F.avg_pool2d(x, factor, factor)


def aligned_noise_pyramid(reference: torch.Tensor, seed: int) -> dict[int, torch.Tensor]:
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    eps32 = torch.randn(reference.shape, generator=generator, device=reference.device)
    # Multiplication by the spatial factor restores unit variance after average pooling.
    return {
        32: eps32,
        16: F.avg_pool2d(eps32, 2, 2) * 2.0,
        8: F.avg_pool2d(eps32, 4, 4) * 4.0,
    }


def infer_num_classes(weights: dict[str, torch.Tensor]) -> int:
    embedding = weights.get("y_embedder.embedding_table.weight")
    return 0 if embedding is None else int(embedding.shape[0] - 1)


def load_checkpoint(path: Path) -> tuple[dict[str, torch.Tensor], Any]:
    try:
        checkpoint = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    weights = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    return weights, checkpoint


def load_unified(path: Path, device: torch.device, dtype: torch.dtype):
    weights, checkpoint = load_checkpoint(path)
    num_classes = infer_num_classes(weights)
    repa_dim = None
    repa_projector_dim = 2048
    if "repa_projector.2.weight" in weights:
        repa_dim = int(weights["repa_projector.2.weight"].shape[0])
        repa_projector_dim = int(weights["repa_projector.2.weight"].shape[1])
    model = DiTMultiRes_models["DiT-XL/2"](
        num_classes=num_classes,
        base_size=16,
        repa_dim=repa_dim,
        repa_projector_dim=repa_projector_dim,
    )
    model.load_state_dict(weights)
    model.to(device=device, dtype=dtype).eval().requires_grad_(False)
    del checkpoint, weights
    gc.collect()
    return model, num_classes


def load_multistage(
    directory: Path, device: torch.device, dtype: torch.dtype
) -> tuple[dict[int, torch.nn.Module], int]:
    architectures = {8: "DiT-B/2", 16: "DiT-L/2", 32: "DiT-XL/2"}
    models: dict[int, torch.nn.Module] = {}
    num_classes = None
    for resolution in (8, 16, 32):
        path = directory / f"checkpoints_{resolution}x{resolution}" / "0400000.pt"
        weights, checkpoint = load_checkpoint(path)
        stage_classes = infer_num_classes(weights)
        if num_classes is None:
            num_classes = stage_classes
        elif num_classes != stage_classes:
            raise ValueError("multistage class-count mismatch")
        d_prediction = "d_final_layer.linear.weight" in weights
        model = DiT_models[architectures[resolution]](
            input_size=resolution,
            num_classes=stage_classes,
            d_prediction=d_prediction,
        )
        model.load_state_dict(weights)
        model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        models[resolution] = model
        del checkpoint, weights
        gc.collect()
    assert num_classes is not None
    return models, num_classes


def make_diffusions() -> dict[int, Any]:
    diffusions = {}
    for index, resolution in enumerate(RESOLUTIONS):
        c = index if index < len(RESOLUTIONS) - 1 else None
        # Both transitions are adjacent 2x stages.  Keep the factory default
        # instead of passing sample_scale explicitly so this probe also works
        # with the exact older diffusion factory used to train these checkpoints.
        diffusions[resolution] = create_diffusion_with_drift(
            timestep_respacing="",
            diffusion_steps=1000,
            c=c,
            T_c=BOUNDARIES[index],
            T_c_plus_1=BOUNDARIES[index + 1],
        )
    return diffusions


def stage_drift(diffusion: Any, clean: torch.Tensor) -> torch.Tensor | None:
    if diffusion.c is None:
        return None
    return diffusion.upsample(
        diffusion.downsample(clean, diffusion.sample_scale), diffusion.sample_scale
    ) - clean


class AttentionCapture:
    def __init__(self, model: torch.nn.Module, layer_ids: list[int], unified: bool):
        self.data: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
        self.tag: str | None = None
        self.unified = unified
        self.handles = []
        for layer in layer_ids:
            module = model.blocks[layer].attn

            def hook(attn_module, args, *, layer_id=layer):
                if self.tag is None:
                    return
                x = args[0]
                batch, seq, width = x.shape
                heads = attn_module.num_heads
                head_dim = width // heads
                qkv = attn_module.qkv(x).reshape(batch, seq, 3, heads, head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q_pre, k_pre, v = qkv.unbind(0)
                if self.unified:
                    q = apply_rotary_emb(q_pre, args[1], args[2])
                    k = apply_rotary_emb(k_pre, args[1], args[2])
                else:
                    q = attn_module.q_norm(q_pre)
                    k = attn_module.k_norm(k_pre)
                scale = float(getattr(attn_module, "scale", head_dim**-0.5))
                attention = (q @ k.transpose(-2, -1) * scale).softmax(-1)
                self.data.setdefault(self.tag, {})[layer_id] = {
                    "hidden": x.detach().cpu().to(torch.float16),
                    "q_pre": q_pre.detach().cpu().to(torch.float16),
                    "k_pre": k_pre.detach().cpu().to(torch.float16),
                    "q": q.detach().cpu().to(torch.float16),
                    "k": k.detach().cpu().to(torch.float16),
                    "v": v.detach().cpu().to(torch.float16),
                    "attention": attention.detach().cpu().to(torch.float16),
                }

            self.handles.append(module.register_forward_pre_hook(hook))

    def close(self):
        for handle in self.handles:
            handle.remove()


def layer_slots(depth: int) -> list[int]:
    return sorted({round(fraction * (depth - 1)) for fraction in SLOT_FRACTIONS})


def root_mapping(grid: int, tile: int) -> torch.Tensor:
    y, x = torch.meshgrid(torch.arange(grid), torch.arange(grid), indexing="ij")
    roots_per_axis = math.ceil(grid / tile)
    return ((y // tile) * roots_per_axis + (x // tile)).flatten().long()


def pool_tokens(x: torch.Tensor, grid: int, tile: int) -> torch.Tensor:
    """Pool [B,H,S,D] or [B,S,D] to corresponding 2-D root cells."""
    squeezed = False
    if x.ndim == 3:
        x = x[:, None]
        squeezed = True
    batch, heads, _, dim = x.shape
    roots_axis = grid // tile
    x = x.reshape(batch, heads, roots_axis, tile, roots_axis, tile, dim).mean((3, 5))
    x = x.reshape(batch, heads, roots_axis * roots_axis, dim)
    return x[:, 0] if squeezed else x


def aggregate_attention(attention: torch.Tensor, grid: int, tile: int) -> torch.Tensor:
    attention = attention.float()
    mapping = root_mapping(grid, tile)
    roots = int(mapping.max()) + 1
    membership = F.one_hot(mapping, roots).float()
    key_mass = attention @ membership
    out = torch.zeros(
        attention.shape[0], attention.shape[1], roots, roots, dtype=attention.dtype
    )
    for root in range(roots):
        out[:, :, root] = key_mass[:, :, mapping == root].mean(2)
    return out / out.sum(-1, keepdim=True).clamp_min(1e-12)


def pooled_route(q: torch.Tensor, k: torch.Tensor, grid: int, tile: int) -> torch.Tensor:
    q_root = pool_tokens(q.float(), grid, tile)
    k_root = pool_tokens(k.float(), grid, tile)
    return (q_root @ k_root.transpose(-2, -1) / math.sqrt(q.shape[-1])).softmax(-1)


def normalize_rows(x: torch.Tensor) -> torch.Tensor:
    return x.clamp_min(0) / x.sum(-1, keepdim=True).clamp_min(1e-12)


def distribution_metrics(
    source: torch.Tensor,
    target: torch.Tensor,
    *,
    force_head_average: bool = False,
) -> dict[str, Any]:
    source = normalize_rows(source.float())
    target = normalize_rows(target.float())
    # Independently trained Multistage models have no head-to-head identity even
    # when two stages happen to use the same number of heads.  In that case only
    # the head-averaged spatial topology is a meaningful cross-model quantity.
    if force_head_average or source.shape[1] != target.shape[1]:
        source_cmp = source.mean(1)
        target_cmp = target.mean(1)
        head_mode = "head-averaged"
    else:
        source_cmp, target_cmp = source, target
        head_mode = "head-matched"
    cosine = F.cosine_similarity(source_cmp, target_cmp, dim=-1)
    midpoint = 0.5 * (source_cmp + target_cmp)
    js = 0.5 * (
        (source_cmp * (source_cmp.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
        + (target_cmp * (target_cmp.clamp_min(1e-12).log() - midpoint.clamp_min(1e-12).log())).sum(-1)
    )
    roots = source_cmp.shape[-1]
    sweep = []
    for density in (0.1, 0.2, 0.3, 0.4):
        keep = max(1, math.ceil(density * roots))
        source_index = source_cmp.topk(keep, dim=-1).indices
        target_index = target_cmp.topk(keep, dim=-1).indices
        source_mask = torch.zeros_like(source_cmp, dtype=torch.bool).scatter_(-1, source_index, True)
        target_mask = torch.zeros_like(target_cmp, dtype=torch.bool).scatter_(-1, target_index, True)
        sweep.append(
            {
                "density": density,
                "effective_density": keep / roots,
                "target_mass_coverage": float((target_cmp * source_mask).sum(-1).mean()),
                "oracle_mass_coverage": float((target_cmp * target_mask).sum(-1).mean()),
                "topk_recall": float((source_mask & target_mask).sum(-1).float().mean() / keep),
            }
        )
    return {
        "head_mode": head_mode,
        "cosine": float(cosine.mean()),
        "cosine_p10": float(torch.quantile(cosine.flatten(), 0.1)),
        "js": float(js.mean()),
        "density_sweep": sweep,
    }


def candidate_refinement_metrics(
    source: torch.Tensor, target_exact: torch.Tensor, target_proxy: torch.Tensor
) -> dict[str, float]:
    source = normalize_rows(source.float().mean(1))
    target = normalize_rows(target_exact.float().mean(1))
    proxy = normalize_rows(target_proxy.float().mean(1))
    roots = source.shape[-1]
    candidate_keep = max(1, math.ceil(0.4 * roots))
    final_keep = max(1, math.ceil(0.2 * roots))
    candidate_idx = source.topk(candidate_keep, dim=-1).indices
    candidate = torch.zeros_like(source, dtype=torch.bool).scatter_(-1, candidate_idx, True)
    restricted_proxy = proxy.masked_fill(~candidate, -torch.inf)
    refined_idx = restricted_proxy.topk(final_keep, dim=-1).indices
    refined = torch.zeros_like(source, dtype=torch.bool).scatter_(-1, refined_idx, True)
    proxy_idx = proxy.topk(final_keep, dim=-1).indices
    proxy_mask = torch.zeros_like(source, dtype=torch.bool).scatter_(-1, proxy_idx, True)
    source_idx = source.topk(final_keep, dim=-1).indices
    source_mask = torch.zeros_like(source, dtype=torch.bool).scatter_(-1, source_idx, True)
    return {
        "candidate_density": candidate_keep / roots,
        "final_density": final_keep / roots,
        "direct_source_coverage": float((target * source_mask).sum(-1).mean()),
        "candidate_plus_fine_proxy_coverage": float((target * refined).sum(-1).mean()),
        "ordinary_fine_proxy_coverage": float((target * proxy_mask).sum(-1).mean()),
    }


def sparse_output_metrics(
    source_route: torch.Tensor,
    target_trace: dict[str, torch.Tensor],
    target_grid: int,
    target_tile: int,
    *,
    force_head_average: bool = False,
) -> dict[str, dict[str, float]]:
    q = target_trace["q"][:2, :4].float()
    k = target_trace["k"][:2, :4].float()
    v = target_trace["v"][:2, :4].float()
    route = source_route[:2].float()
    if force_head_average or route.shape[1] != q.shape[1]:
        route = route.mean(1, keepdim=True).expand(-1, q.shape[1], -1, -1)
    mapping = root_mapping(target_grid, target_tile)
    logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    dense = logits.softmax(-1) @ v
    roots = route.shape[-1]
    result = {}
    for density in (0.2, 0.3, 0.4):
        keep = max(1, math.ceil(density * roots))
        allowed_root = torch.zeros_like(route, dtype=torch.bool).scatter_(
            -1, route.topk(keep, dim=-1).indices, True
        )
        token_allowed = allowed_root[:, :, mapping][:, :, :, mapping]
        sparse = logits.masked_fill(~token_allowed, -torch.inf).softmax(-1) @ v
        result[f"{int(density * 100)}pct"] = {
            "effective_density": keep / roots,
            "output_cosine": float(F.cosine_similarity(sparse, dense, dim=-1).mean()),
            "relative_l2": float((sparse - dense).norm() / dense.norm().clamp_min(1e-12)),
        }
    return result


def summary_tail_metrics(
    source_route: torch.Tensor,
    source_trace: dict[str, torch.Tensor],
    source_grid: int,
    target_trace: dict[str, torch.Tensor],
    target_grid: int,
) -> dict[str, dict[str, float]] | None:
    if source_trace["k"].shape[1] != target_trace["k"].shape[1] or source_trace["k"].shape[-1] != target_trace["k"].shape[-1]:
        return None
    q = target_trace["q"][:2, :4].float()
    k = target_trace["k"][:2, :4].float()
    v = target_trace["v"][:2, :4].float()
    route = source_route[:2, :4].float()
    target_mapping = root_mapping(target_grid, 2)
    roots = route.shape[-1]
    keep = max(1, math.ceil(0.2 * roots))
    selected = torch.zeros_like(route, dtype=torch.bool).scatter_(
        -1, route.topk(keep, dim=-1).indices, True
    )
    source_k = pool_tokens(source_trace["k"][:2, :4].float(), source_grid, 1)
    source_v = pool_tokens(source_trace["v"][:2, :4].float(), source_grid, 1)
    target_k = pool_tokens(k, target_grid, 2)
    target_v = pool_tokens(v, target_grid, 2)
    logits = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    dense = logits.softmax(-1) @ v
    q_roots = target_mapping
    selected_for_tokens = selected[:, :, q_roots]
    exact_allowed = selected_for_tokens[:, :, :, target_mapping]
    exact_logits = logits.masked_fill(~exact_allowed, -torch.inf)
    counts = torch.bincount(target_mapping, minlength=roots).float().clamp_min(1).log()

    def evaluate(summary_k, summary_v):
        tail_logits = q @ summary_k.transpose(-2, -1) / math.sqrt(q.shape[-1])
        tail_logits = tail_logits + counts[None, None, None]
        tail_logits = tail_logits.masked_fill(selected_for_tokens, -torch.inf)
        weights = torch.cat((exact_logits, tail_logits), dim=-1).softmax(-1)
        output = weights[..., : k.shape[2]] @ v + weights[..., k.shape[2] :] @ summary_v
        return {
            "output_cosine": float(F.cosine_similarity(output, dense, dim=-1).mean()),
            "relative_l2": float((output - dense).norm() / dense.norm().clamp_min(1e-12)),
        }

    return {
        "drop_tail": sparse_output_metrics(source_route, target_trace, target_grid, 2)["20pct"],
        "previous_stage_summary": evaluate(source_k, source_v),
        "current_fine_summary_upper_bound": evaluate(target_k, target_v),
    }


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.float() - x.float().mean(0, keepdim=True)
    y = y.float() - y.float().mean(0, keepdim=True)
    k = x @ x.T
    l = y @ y.T
    n = k.shape[0]
    centering = torch.eye(n) - torch.ones(n, n) / n
    kc, lc = centering @ k @ centering, centering @ l @ centering
    return float((kc * lc).sum() / ((kc.square().sum() * lc.square().sum()).sqrt().clamp_min(1e-12)))


def representation_metrics(
    source: torch.Tensor, source_grid: int, target: torch.Tensor, target_grid: int
) -> dict[str, float]:
    source_root = pool_tokens(source.float(), source_grid, 1)
    target_root = pool_tokens(target.float(), target_grid, 2)
    batch, roots, _ = source_root.shape
    flat_x, flat_y = source_root.flatten(0, 1), target_root.flatten(0, 1)
    result = {"linear_cka": linear_cka(flat_x, flat_y)}
    if flat_x.shape[-1] == flat_y.shape[-1]:
        result["aligned_token_cosine"] = float(F.cosine_similarity(flat_x, flat_y, dim=-1).mean())

    # Kernel ridge adapter: fit on half the images and test on unseen images.
    split = max(1, batch // 2)
    train_x, test_x = source_root[:split].flatten(0, 1), source_root[split:].flatten(0, 1)
    train_y, test_y = target_root[:split].flatten(0, 1), target_root[split:].flatten(0, 1)
    if test_x.numel() > 0:
        mean_x, mean_y = train_x.mean(0, keepdim=True), train_y.mean(0, keepdim=True)
        train_x, test_x = train_x - mean_x, test_x - mean_x
        train_y, test_y = train_y - mean_y, test_y - mean_y
        kernel = train_x @ train_x.T
        ridge = 1e-2 * kernel.diag().mean().clamp_min(1e-6)
        alpha = torch.linalg.solve(kernel + ridge * torch.eye(kernel.shape[0]), train_y)
        prediction = test_x @ train_x.T @ alpha
        result["ridge_adapter_test_cosine"] = float(
            F.cosine_similarity(prediction, test_y, dim=-1).mean()
        )
        result["ridge_adapter_relative_l2"] = float(
            (prediction - test_y).norm() / test_y.norm().clamp_min(1e-12)
        )
    return result


def tensor_consistency(source: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    target_low = F.avg_pool2d(target.float(), target.shape[-1] // source.shape[-1])
    source = source.float()
    return {
        "cosine": float(F.cosine_similarity(source.flatten(1), target_low.flatten(1), dim=-1).mean()),
        "relative_l2": float((source - target_low).norm() / target_low.norm().clamp_min(1e-12)),
    }


@torch.inference_mode()
def run_seed(
    model_kind: str,
    models: dict[int, torch.nn.Module],
    captures: dict[int, AttentionCapture],
    diffusions: dict[int, Any],
    clean: dict[int, torch.Tensor],
    labels: torch.Tensor,
    seed: int,
) -> tuple[dict[str, Any], dict[str, dict[int, dict[str, torch.Tensor]]]]:
    noise = aligned_noise_pyramid(clean[32], seed)
    call_specs = {
        "8_end": (8, 0),
        "16_start": (16, STAGE_LENGTHS[1] - 1),
        "16_end": (16, 0),
        "32_start": (32, STAGE_LENGTHS[0] - 1),
    }
    calls = {}
    traces = {}
    for tag, (resolution, local_timestep) in call_specs.items():
        diffusion = diffusions[resolution]
        local_t = torch.full(
            (clean[resolution].shape[0],), local_timestep, device=clean[resolution].device, dtype=torch.long
        )
        x_t = diffusion.q_sample(
            clean[resolution],
            local_t,
            d=stage_drift(diffusion, clean[resolution]),
            noise=noise[resolution],
        )
        capture = captures[resolution]
        capture.tag = tag
        # Timestep sinusoidal features are deliberately formed in fp32 by DiT.
        # Autocast is therefore required when the checkpoint weights are bf16;
        # this matches the mixed-precision path used by the normal sampler.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            output = diffusion.p_mean_variance(
                models[resolution],
                x_t.to(next(models[resolution].parameters()).dtype),
                local_t,
                clip_denoised=False,
                model_kwargs={"y": labels},
            )
        capture.tag = None
        calls[tag] = {
            "resolution": resolution,
            "local_timestep": local_timestep,
            "global_timestep": int(diffusion.timestep_map[local_timestep]),
            "input": x_t.detach().cpu(),
            "pred_x0": output["pred_xstart"].detach().float().cpu(),
        }
        traces[tag] = capture.data.pop(tag)
    return calls, traces


def analyze_pair(
    source_tag: str,
    target_tag: str,
    source_resolution: int,
    target_resolution: int,
    source_layers: list[int],
    target_layers: list[int],
    calls: dict[str, Any],
    traces: dict[str, dict[int, dict[str, torch.Tensor]]],
    unified: bool,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "input_lowpass_consistency": tensor_consistency(
            calls[source_tag]["input"], calls[target_tag]["input"]
        ),
        "pred_x0_lowpass_consistency": tensor_consistency(
            calls[source_tag]["pred_x0"], calls[target_tag]["pred_x0"]
        ),
        "layers": {},
    }
    source_grid, target_grid = source_resolution // 2, target_resolution // 2
    for slot, (source_layer, target_layer) in enumerate(zip(source_layers, target_layers)):
        source = traces[source_tag][source_layer]
        target = traces[target_tag][target_layer]
        source_exact = aggregate_attention(source["attention"], source_grid, 1)
        target_exact = aggregate_attention(target["attention"], target_grid, 2)
        target_proxy = pooled_route(target["q"], target["k"], target_grid, 2)
        layer_result: dict[str, Any] = {
            "source_layer": source_layer,
            "target_layer": target_layer,
            "route": distribution_metrics(
                source_exact,
                target_exact,
                force_head_average=not unified,
            ),
            "candidate_refinement": candidate_refinement_metrics(
                source_exact, target_exact, target_proxy
            ),
            "sparse_fine_output": sparse_output_metrics(
                source_exact,
                target,
                target_grid,
                2,
                force_head_average=not unified,
            ),
            "hidden": representation_metrics(
                source["hidden"], source_grid, target["hidden"], target_grid
            ),
        }
        if unified:
            source_content = pooled_route(source["q_pre"], source["k_pre"], source_grid, 1)
            layer_result["content_only_route"] = distribution_metrics(
                source_content, target_exact
            )
            layer_result["k_root"] = representation_metrics(
                source["k"].permute(0, 2, 1, 3).flatten(2),
                source_grid,
                target["k"].permute(0, 2, 1, 3).flatten(2),
                target_grid,
            )
            layer_result["summary_tail"] = summary_tail_metrics(
                source_exact, source, source_grid, target, target_grid
            )
        result["layers"][str(slot)] = layer_result
    return result


def aggregate_seed_reports(seed_reports: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate = {}
    for model_kind in ("unified", "multistage"):
        aggregate[model_kind] = {}
        for pair in ("8_to_16", "16_to_32"):
            route_cosines, route_js, coverages, sparse_cosines = [], [], [], []
            pred_cosines = []
            summary_parent, summary_current = [], []
            for report in seed_reports:
                pair_report = report[model_kind][pair]
                pred_cosines.append(pair_report["pred_x0_lowpass_consistency"]["cosine"])
                for layer in pair_report["layers"].values():
                    route_cosines.append(layer["route"]["cosine"])
                    route_js.append(layer["route"]["js"])
                    row = next(
                        item for item in layer["route"]["density_sweep"] if item["density"] == 0.2
                    )
                    coverages.append(row["target_mass_coverage"])
                    sparse_cosines.append(layer["sparse_fine_output"]["20pct"]["output_cosine"])
                    if layer.get("summary_tail"):
                        summary_parent.append(
                            layer["summary_tail"]["previous_stage_summary"]["output_cosine"]
                        )
                        summary_current.append(
                            layer["summary_tail"]["current_fine_summary_upper_bound"]["output_cosine"]
                        )
            aggregate[model_kind][pair] = {
                "pred_x0_lowpass_cosine": float(np.mean(pred_cosines)),
                "route_cosine": float(np.mean(route_cosines)),
                "route_js": float(np.mean(route_js)),
                "route_20pct_mass_coverage": float(np.mean(coverages)),
                "direct_20pct_output_cosine": float(np.mean(sparse_cosines)),
                "parent_summary_20pct_output_cosine": float(np.mean(summary_parent)) if summary_parent else None,
                "current_summary_20pct_output_cosine": float(np.mean(summary_current)) if summary_current else None,
            }
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--unified-ckpt", required=True)
    parser.add_argument("--multistage-dir", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--imagenet-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seeds", default="1234,5678")
    parser.add_argument("--memory-fraction", type=float, default=0.24)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the XL checkpoint probe")
    torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
    dtype = torch.bfloat16
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths, labels = discover_images(Path(args.imagenet_root), args.batch_size)
    labels = labels.to(device)
    latent32 = encode_images(paths, Path(args.vae_path), device)
    clean = {resolution: downsample_latent(latent32, resolution) for resolution in RESOLUTIONS}
    diffusions = make_diffusions()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    report: dict[str, Any] = {
        "scope": "trained RDD diffusion checkpoints on deterministic ImageNet VAE latents",
        "stage_lengths_finest_to_coarsest": list(STAGE_LENGTHS),
        "boundaries": list(BOUNDARIES),
        "images": [str(path) for path in paths],
        "labels": labels.cpu().tolist(),
        "noise_seeds": seeds,
        "unified_checkpoint": str(Path(args.unified_ckpt).resolve()),
        "multistage_directory": str(Path(args.multistage_dir).resolve()),
        "per_seed": [],
    }

    # Unified XL: one shared model, direct hidden/K/V transfer is dimensionally valid.
    unified_model, unified_classes = load_unified(Path(args.unified_ckpt), device, dtype)
    if unified_classes != 1000:
        raise ValueError(f"unexpected unified class count: {unified_classes}")
    unified_layers = layer_slots(len(unified_model.blocks))
    unified_capture = AttentionCapture(unified_model, unified_layers, unified=True)
    unified_models = {resolution: unified_model for resolution in RESOLUTIONS}
    unified_captures = {resolution: unified_capture for resolution in RESOLUTIONS}
    unified_runs = []
    for seed in seeds:
        calls, traces = run_seed(
            "unified", unified_models, unified_captures, diffusions, clean, labels, seed
        )
        unified_runs.append(
            {
                "calls": calls,
                "8_to_16": analyze_pair(
                    "8_end", "16_start", 8, 16,
                    unified_layers, unified_layers, calls, traces, unified=True,
                ),
                "16_to_32": analyze_pair(
                    "16_end", "32_start", 16, 32,
                    unified_layers, unified_layers, calls, traces, unified=True,
                ),
            }
        )
    unified_capture.close()
    del unified_model, unified_models, unified_captures
    gc.collect()
    torch.cuda.empty_cache()

    # Multistage XL+L+B: topology can transfer, hidden/K/V require adapters.
    multistage_models, multistage_classes = load_multistage(
        Path(args.multistage_dir), device, dtype
    )
    if multistage_classes != 1000:
        raise ValueError(f"unexpected multistage class count: {multistage_classes}")
    multistage_layers = {
        resolution: layer_slots(len(model.blocks)) for resolution, model in multistage_models.items()
    }
    multistage_captures = {
        resolution: AttentionCapture(model, multistage_layers[resolution], unified=False)
        for resolution, model in multistage_models.items()
    }
    multistage_runs = []
    for seed in seeds:
        calls, traces = run_seed(
            "multistage", multistage_models, multistage_captures,
            diffusions, clean, labels, seed,
        )
        multistage_runs.append(
            {
                "calls": calls,
                "8_to_16": analyze_pair(
                    "8_end", "16_start", 8, 16,
                    multistage_layers[8], multistage_layers[16],
                    calls, traces, unified=False,
                ),
                "16_to_32": analyze_pair(
                    "16_end", "32_start", 16, 32,
                    multistage_layers[16], multistage_layers[32],
                    calls, traces, unified=False,
                ),
            }
        )
    for capture in multistage_captures.values():
        capture.close()

    for seed_index, seed in enumerate(seeds):
        report["per_seed"].append(
            {
                "seed": seed,
                "unified": unified_runs[seed_index],
                "multistage": multistage_runs[seed_index],
            }
        )
    compact = [
        {
            "unified": {pair: run[pair] for pair in ("8_to_16", "16_to_32")},
            "multistage": {
                pair: multistage_runs[index][pair] for pair in ("8_to_16", "16_to_32")
            },
        }
        for index, run in enumerate(unified_runs)
    ]
    report["aggregate"] = aggregate_seed_reports(compact)

    # Tensor-valued call inputs/predictions are diagnostic intermediates, not JSON output.
    for seed_report in report["per_seed"]:
        for model_kind in ("unified", "multistage"):
            seed_report[model_kind].pop("calls", None)
    path = output / "rdd_2d_cross_stage_probe.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["aggregate"], indent=2), flush=True)
    print(f"saved: {path}", flush=True)


if __name__ == "__main__":
    main()
