#!/usr/bin/env python3
"""Profile per-stage local solver error for RDD equal-FLOPs allocation.

For the same teacher-forced noisy state, this compares one update spanning h
with two updates spanning h/2.  The squared discrepancy, normalized by the
expected step-size power, is a cheap local-error-density proxy.  It is not FID;
its purpose is to reduce the integer allocation search to a few candidates.

All topology/checkpoint arguments use finest -> coarsest order.  The emitted
``difficulty_sampling_order`` uses coarse -> fine order and can be passed
directly to ``rdd_equal_flops_allocation.py --difficulty``.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder


ROOT = Path(__file__).resolve().parents[2]
DIT_PATH = ROOT / "DiT"
sys.path.insert(0, str(DIT_PATH))
from models import DiT_models  # noqa: E402
sys.path.remove(str(DIT_PATH))
sys.path.insert(0, str(ROOT))
from diffusion import create_diffusion_with_drift  # noqa: E402


DEFAULT_ARCHITECTURES = {32: "DiT-XL/2", 16: "DiT-L/2", 8: "DiT-B/2"}
PIL_RESAMPLING = getattr(Image, "Resampling", Image)


def csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def csv_strings(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def center_crop_arr(pil_image: Image.Image, image_size: int) -> Image.Image:
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(value // 2 for value in pil_image.size), resample=PIL_RESAMPLING.BOX
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(value * scale) for value in pil_image.size),
        resample=PIL_RESAMPLING.BICUBIC,
    )
    array = np.array(pil_image)
    crop_y = (array.shape[0] - image_size) // 2
    crop_x = (array.shape[1] - image_size) // 2
    return Image.fromarray(
        array[crop_y : crop_y + image_size, crop_x : crop_x + image_size]
    )


def downsample(latent: torch.Tensor, resolution: int) -> torch.Tensor:
    factor = latent.shape[-1] // resolution
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)


def infer_num_classes(weights: dict[str, torch.Tensor]) -> int:
    embedding = weights.get("y_embedder.embedding_table.weight")
    return 0 if embedding is None else int(embedding.shape[0] - 1)


def load_model(checkpoint: Path, architecture: str, resolution: int, device: torch.device):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    weights = state.get("ema", state.get("model"))
    if weights is None:
        raise ValueError(f"checkpoint has neither ema nor model weights: {checkpoint}")
    classes = infer_num_classes(weights)
    d_prediction = "d_final_layer.linear.weight" in weights
    model = DiT_models[architecture](
        input_size=resolution, num_classes=classes, d_prediction=d_prediction
    )
    model.load_state_dict(weights)
    model.to(device).eval()
    del state, weights
    return model, classes


def solver_update(diffusion, model, x, t, t_next, y, solver: str):
    kwargs = {"model_kwargs": {"y": y}, "clip_denoised": False}
    if solver == "ddpm":
        return diffusion.jump_step_sample(
            model.forward, x, t, t_next, deterministic=True, **kwargs
        )["sample"]
    if solver == "dpm1":
        return diffusion.dpm_solver_sample(
            model.forward, x, t, t_next, order=1, **kwargs
        )["sample"]
    if solver == "dpm2":
        return diffusion.dpm_solver_sample(
            model.forward, x, t, t_next, order=2, **kwargs
        )["sample"]
    raise ValueError(solver)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--ckpt-step", default="0400000")
    parser.add_argument("--resolutions", default="32,16,8")
    parser.add_argument("--stage-lengths", default="300,300,400")
    parser.add_argument("--architectures", default="DiT-XL/2,DiT-L/2,DiT-B/2")
    parser.add_argument("--solver", choices=["ddpm", "dpm1", "dpm2"], default="ddpm")
    parser.add_argument("--num-images", type=int, default=256)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument("--probe-bins", type=int, default=8)
    parser.add_argument(
        "--half-span-fraction", type=float, default=0.01,
        help="Each half update spans this fraction of the stage interval.",
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    resolutions = csv_ints(args.resolutions)
    lengths = csv_ints(args.stage_lengths)
    architectures = csv_strings(args.architectures)
    if not len(resolutions) == len(lengths) == len(architectures):
        raise ValueError("resolutions, stage-lengths and architectures must align")
    if sum(lengths) != 1000 or any(value <= 0 for value in lengths):
        raise ValueError("stage-lengths must be positive and sum to 1000")
    if args.num_images <= 0 or args.probe_bins <= 0:
        raise ValueError("num-images and probe-bins must be positive")
    if not 0 < args.half_span_fraction < 0.25:
        raise ValueError("half-span-fraction must be in (0, 0.25)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image.convert("RGB"), 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    full_dataset = ImageFolder(str(args.data_path), transform=transform)
    subset = Subset(full_dataset, range(min(args.num_images, len(full_dataset))))
    loader = DataLoader(
        subset, batch_size=args.image_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    vae = AutoencoderKL.from_pretrained(
        str(args.vae_path), local_files_only=True
    ).to(device).eval()
    latent_batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    with torch.no_grad():
        for images, labels in loader:
            latent = vae.encode(images.to(device, non_blocking=True)).latent_dist.mean
            latent_batches.append((latent.mul(0.18215).cpu(), labels.cpu()))
    del vae
    torch.cuda.empty_cache()

    boundaries = [0]
    for length in lengths:
        boundaries.append(boundaries[-1] + length)
    stage_records = []
    solver_order = 2 if args.solver == "dpm2" else 1

    for index, (resolution, length, architecture) in enumerate(
        zip(resolutions, lengths, architectures)
    ):
        checkpoint = args.ckpt_dir / f"checkpoints_{resolution}x{resolution}" / f"{args.ckpt_step}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, num_classes = load_model(checkpoint, architecture, resolution, device)
        projection_scale = (
            resolution // resolutions[index + 1] if index < len(resolutions) - 1 else 2
        )
        diffusion = create_diffusion_with_drift(
            "", diffusion_steps=1000,
            c=index if index < len(resolutions) - 1 else None,
            T_c=boundaries[index], T_c_plus_1=boundaries[index + 1],
        )
        half_span = max(1, round(length * args.half_span_fraction))
        maximum_start = length - 1
        minimum_start = 2 * half_span
        if minimum_start > maximum_start:
            raise ValueError(f"stage {resolution} is too short for step doubling")
        probe_times = sorted(set(
            int(round(value)) for value in np.linspace(
                minimum_start, maximum_start, args.probe_bins
            )
        ))
        normalized_full_span = (2.0 * half_span) / length
        normalization_power = 2 * (solver_order + 1)
        bin_sums = {time_index: 0.0 for time_index in probe_times}
        bin_elements = {time_index: 0 for time_index in probe_times}

        with torch.no_grad():
            for latent_cpu, labels_cpu in latent_batches:
                clean = downsample(latent_cpu.to(device), resolution)
                labels = labels_cpu.to(device)
                if num_classes <= 0:
                    labels = torch.zeros_like(labels)
                elif labels.max().item() >= num_classes:
                    raise ValueError("dataset label exceeds checkpoint class count")
                drift = None
                if index < len(resolutions) - 1:
                    low = F.avg_pool2d(
                        clean, kernel_size=projection_scale, stride=projection_scale
                    )
                    drift = low.repeat_interleave(projection_scale, -2).repeat_interleave(
                        projection_scale, -1
                    ) - clean
                for time_index in probe_times:
                    t = torch.full(
                        (clean.shape[0],), time_index, device=device, dtype=torch.long
                    )
                    t_middle = t - half_span
                    t_next = t - 2 * half_span
                    noise = torch.randn_like(clean)
                    state = diffusion.q_sample(clean, t, d=drift, noise=noise)
                    one = solver_update(
                        diffusion, model, state, t, t_next, labels, args.solver
                    )
                    half = solver_update(
                        diffusion, model, state, t, t_middle, labels, args.solver
                    )
                    two = solver_update(
                        diffusion, model, half, t_middle, t_next, labels, args.solver
                    )
                    difference = (one - two).float().square()
                    bin_sums[time_index] += difference.sum().item()
                    bin_elements[time_index] += difference.numel()

        raw_by_bin = [
            bin_sums[time_index] / bin_elements[time_index]
            for time_index in probe_times
        ]
        normalized_by_bin = [
            value / (normalized_full_span**normalization_power)
            for value in raw_by_bin
        ]
        density = float(np.mean(normalized_by_bin))
        stage_records.append({
            "resolution": resolution,
            "architecture": architecture,
            "checkpoint": str(checkpoint),
            "interval_global": [boundaries[index], boundaries[index + 1]],
            "sample_scale": projection_scale if index < len(resolutions) - 1 else None,
            "probe_local_timesteps": probe_times,
            "half_span_local_timesteps": half_span,
            "normalized_full_span": normalized_full_span,
            "raw_step_doubling_mse_by_bin": raw_by_bin,
            "normalized_error_density_by_bin": normalized_by_bin,
            "difficulty": density,
        })
        print(f"{resolution}x{resolution}: difficulty={density:.8g}")
        del model, diffusion
        gc.collect()
        torch.cuda.empty_cache()

    records_sampling_order = list(reversed(stage_records))
    difficulties = [record["difficulty"] for record in records_sampling_order]
    # Ratios are easier to inspect and numerically equivalent in the allocator.
    minimum = min(difficulties)
    difficulty_ratios = [value / minimum for value in difficulties]
    payload = {
        "status": "local_step_doubling_proxy_not_fid",
        "solver": args.solver,
        "solver_order": solver_order,
        "resolutions_finest_to_coarsest": resolutions,
        "stage_lengths_finest_to_coarsest": lengths,
        "sampling_order": [record["resolution"] for record in records_sampling_order],
        "difficulty_sampling_order": difficulties,
        "difficulty_ratio_sampling_order": difficulty_ratios,
        "difficulty_cli": ",".join(f"{value:.8g}" for value in difficulty_ratios),
        "num_images": sum(batch[0].shape[0] for batch in latent_batches),
        "teacher_forced_states": True,
        "stochastic_ddpm_profiled_deterministically": args.solver == "ddpm",
        "stages_finest_to_coarsest": stage_records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"difficulty for allocator (coarse->fine): {payload['difficulty_cli']}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
