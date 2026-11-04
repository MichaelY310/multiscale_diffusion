#!/usr/bin/env python3
"""Sample a distilled Unified RDD model in exactly three conditional NFEs."""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers.models import AutoencoderKL


ROOT = Path(__file__).resolve().parents[1]
DIT_ROOT = ROOT / "DiT"
sys.path.insert(0, str(DIT_ROOT))
from models_multires import DiTMultiRes_models  # noqa: E402
sys.path.remove(str(DIT_ROOT))
sys.path.insert(0, str(ROOT))

from diffusion import create_diffusion_with_drift  # noqa: E402
from diffusion.rdd_phased_dmd import (  # noqa: E402
    fresh_transition,
    sampling_phases,
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def infer_num_classes(weights: dict[str, torch.Tensor]) -> int:
    for key, value in weights.items():
        if key.endswith("y_embedder.embedding_table.weight"):
            return int(value.shape[0]) - 1
    raise KeyError("could not infer num_classes")


def load_model(
    checkpoint_path: str,
    architecture: str,
    resolutions,
    device,
    weight_source: str = "auto",
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if weight_source == "auto":
        source_order = ("ema", "student", "model")
    else:
        source_order = (weight_source,)
    weights = None
    selected_source = None
    for source in source_order:
        candidate = checkpoint.get(source)
        if isinstance(candidate, dict):
            weights = candidate
            selected_source = source
            break
    if weights is None and weight_source == "auto" and all(
        torch.is_tensor(value) for value in checkpoint.values()
    ):
        weights = checkpoint
        selected_source = "root"
    if not isinstance(weights, dict):
        available = sorted(key for key, value in checkpoint.items() if isinstance(value, dict))
        raise TypeError(
            f"checkpoint does not contain requested {weight_source!r} weights; "
            f"available state dictionaries: {available}"
        )
    num_classes = infer_num_classes(weights)
    patch_size = int(architecture.split("/")[1])
    repa_dim = None
    repa_projector_dim = 2048
    if "repa_projector.2.weight" in weights:
        repa_dim = int(weights["repa_projector.2.weight"].shape[0])
        repa_projector_dim = int(weights["repa_projector.2.weight"].shape[1])
    model = DiTMultiRes_models[architecture](
        num_classes=num_classes,
        base_size=int(resolutions[0]) // patch_size,
        repa_dim=repa_dim,
        repa_projector_dim=repa_projector_dim,
    )
    model.load_state_dict(weights, strict=True)
    model.to(device).eval()
    print(
        f"loaded weight_source={selected_source}, checkpoint_step={checkpoint.get('step')}",
        flush=True,
    )
    del checkpoint, weights
    gc.collect()
    return model, num_classes


def build_diffusions(stage_lengths, resolutions):
    boundaries = [0]
    for length in stage_lengths:
        boundaries.append(boundaries[-1] + int(length))
    total = sum(stage_lengths)
    result = []
    for index, resolution in enumerate(resolutions):
        scale = resolution // resolutions[index + 1] if index + 1 < len(resolutions) else 2
        result.append(
            create_diffusion_with_drift(
                timestep_respacing="",
                diffusion_steps=total,
                c=index if index + 1 < len(resolutions) else None,
                T_c=boundaries[index],
                T_c_plus_1=boundaries[index + 1],
                sample_scale=scale,
            )
        )
    return result


def predict_clean(diffusion, model, x, labels):
    local_t = torch.full(
        (x.shape[0],),
        int(diffusion.num_timesteps) - 1,
        device=x.device,
        dtype=torch.long,
    )
    return diffusion.p_mean_variance(
        model,
        x,
        local_t,
        clip_denoised=False,
        model_kwargs={"y": labels},
    )["pred_xstart"]


@torch.no_grad()
def sample_three_nfe(model, diffusions, resolutions, labels, bf16: bool):
    phases = sampling_phases(resolutions)
    state = torch.randn(
        labels.shape[0], 4, phases[0].resolution, phases[0].resolution, device=labels.device
    )
    clean = None
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bf16):
        for phase in phases:
            diffusion = diffusions[phase.diffusion_index]
            clean = predict_clean(diffusion, model, state, labels).float()
            if phase.phase_index + 1 < len(phases):
                next_phase = phases[phase.phase_index + 1]
                state = fresh_transition(
                    clean, next_phase.resolution, diffusions[next_phase.diffusion_index]
                )
    return clean


@torch.no_grad()
def decode_and_save(vae, latents, paths, batch_size: int):
    images = []
    for start in range(0, latents.shape[0], int(batch_size)):
        decoded = vae.decode(latents[start : start + batch_size] / 0.18215).sample
        decoded = (decoded / 2 + 0.5).clamp(0, 1)
        decoded = (decoded * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
        images.extend(decoded)
    for array, path in zip(images, paths):
        temporary = f"{path}.tmp-{os.getpid()}.png"
        Image.fromarray(array).save(temporary)
        os.replace(temporary, path)


def main(args):
    if args.rank < 0 or args.rank >= args.world_size:
        raise ValueError("rank must be in [0, world_size)")
    device = torch.device("cuda")
    resolutions = parse_csv_ints(args.resolutions)
    stage_lengths = parse_csv_ints(args.stage_lengths)
    if len(resolutions) != 3 or len(stage_lengths) != 3:
        raise ValueError("three-step sampling requires exactly three stages")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, num_classes = load_model(
        args.checkpoint,
        args.model,
        resolutions,
        device,
        weight_source=args.weight_source,
    )
    diffusions = build_diffusions(stage_lengths, resolutions)
    vae = AutoencoderKL.from_pretrained(args.vae_path, local_files_only=True).to(device).eval()

    indices = list(range(args.rank, args.total_samples, args.world_size))
    pending = [index for index in indices if not (output_dir / f"{index:06d}.png").exists()]
    print(
        f"rank={args.rank}/{args.world_size}, pending={len(pending)}/{len(indices)}, "
        "NFE=3 (8x8 + 16x16 + 32x32), CFG=1 conditional",
        flush=True,
    )
    started = time.time()
    for offset in range(0, len(pending), args.batch_size):
        batch_indices = pending[offset : offset + args.batch_size]
        torch.manual_seed(args.seed + batch_indices[0])
        labels = torch.randint(0, num_classes, (len(batch_indices),), device=device)
        latents = sample_three_nfe(model, diffusions, resolutions, labels, args.bf16)
        paths = [str(output_dir / f"{index:06d}.png") for index in batch_indices]
        decode_and_save(vae, latents, paths, args.vae_batch_size)
        done = min(offset + len(batch_indices), len(pending))
        print(
            f"rank={args.rank} generated={done}/{len(pending)} elapsed={time.time()-started:.1f}s",
            flush=True,
        )
    print(f"rank={args.rank} complete", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vae-path", required=True)
    parser.add_argument("--model", default="DiT-L/2")
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument("--total-samples", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--vae-batch-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--weight-source",
        choices=("auto", "ema", "student", "model"),
        default="auto",
        help="Select a checkpoint state dictionary explicitly; auto keeps legacy EMA-first behavior.",
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
