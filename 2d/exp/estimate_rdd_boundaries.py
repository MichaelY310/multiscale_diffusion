#!/usr/bin/env python3
"""Estimate RDD boundary candidates from native-ImageNet VAE latents.

This is a data-only prior, not a claim of FID optimality.  It measures exactly
the projection residuals used by the implementation and finds the forward
diffusion time at which each residual band's no-drift SNR first falls below a
shared threshold.  The resulting split should seed the short-run boundary
search, and must still be calibrated/validated with real checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from diffusion.gaussian_diffusion_with_drift import get_named_beta_schedule  # noqa: E402

PIL_RESAMPLING = getattr(Image, "Resampling", Image)


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


def project(latent: torch.Tensor, resolution: int) -> torch.Tensor:
    factor = latent.shape[-1] // resolution
    pooled = F.avg_pool2d(latent, kernel_size=factor, stride=factor)
    return pooled.repeat_interleave(factor, -2).repeat_interleave(factor, -1)


def first_snr_below(alpha_bar: np.ndarray, variance: float, threshold: float) -> int:
    snr = alpha_bar * variance / np.maximum(1.0 - alpha_bar, 1e-20)
    indices = np.flatnonzero(snr <= threshold)
    return int(indices[0]) if len(indices) else len(alpha_bar) - 1


def round_boundary(value: int, grid: int) -> int:
    return int(round(value / grid) * grid)


def csv_thresholds(value: str) -> list[float]:
    thresholds = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not thresholds or any(item <= 0 for item in thresholds):
        raise ValueError("thresholds must contain positive comma-separated values")
    return thresholds


def interval_summary(values: list[int], grid: int) -> dict[str, int]:
    array = np.asarray(values, dtype=np.float64)
    low = round_boundary(int(round(np.quantile(array, 0.05))), grid)
    center = round_boundary(int(round(np.median(array))), grid)
    high = round_boundary(int(round(np.quantile(array, 0.95))), grid)
    half_width = max(center - low, high - center)
    radius = int(np.clip(round_boundary(max(half_width, grid), grid), grid, 150))
    return {"q05": low, "median": center, "q95": high, "search_radius": radius}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--num-images", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--posterior", choices=["mean", "sample"], default="mean")
    parser.add_argument("--snr-threshold", type=float, default=1.0)
    parser.add_argument(
        "--thresholds", default="0.5,1.0,2.0",
        help="Threshold-sensitivity values included in the uncertainty interval.",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--grid", type=int, default=50)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if (
        args.num_images <= 0 or args.snr_threshold <= 0 or args.grid <= 0
        or args.bootstrap_replicates < 0
    ):
        raise ValueError(
            "num-images, snr-threshold and grid must be positive; "
            "bootstrap-replicates must be non-negative"
        )
    thresholds = csv_thresholds(args.thresholds)
    if args.snr_threshold not in thresholds:
        thresholds.append(args.snr_threshold)
        thresholds.sort()
    transform = transforms.Compose([
        transforms.Lambda(lambda image: center_crop_arr(image.convert("RGB"), 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageFolder(str(args.data_path), transform=transform)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )
    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(
        str(args.vae_path), local_files_only=True
    ).to(device).eval()

    sums = {"coarse_8": 0.0, "band_16_minus_8": 0.0,
            "band_32_minus_16": 0.0, "band_32_minus_8": 0.0}
    # All residuals are represented on the 32x32 grid, but they occupy
    # subspaces with different ranks.  Comparing mean-square values over all
    # 32x32 pixels would unfairly dilute the narrow 16-minus-8 band.  Divide
    # by each subspace's active degree-of-freedom fraction so signal energy is
    # compared to the amount of isotropic diffusion noise in the same band.
    active_dof_fraction = {
        "coarse_8": 8**2 / 32**2,
        "band_16_minus_8": (16**2 - 8**2) / 32**2,
        "band_32_minus_16": (32**2 - 16**2) / 32**2,
        "band_32_minus_8": (32**2 - 8**2) / 32**2,
    }
    per_image_energy = {name: [] for name in sums}
    elements = 0
    images_seen = 0
    generator = torch.Generator(device=device).manual_seed(0)
    with torch.no_grad():
        for images, _ in loader:
            if images_seen >= args.num_images:
                break
            images = images[: args.num_images - images_seen].to(device, non_blocking=True)
            posterior = vae.encode(images).latent_dist
            if args.posterior == "mean":
                latent = posterior.mean
            else:
                latent = posterior.sample(generator=generator)
            latent = latent * 0.18215
            p16, p8 = project(latent, 16), project(latent, 8)
            bands = {
                "coarse_8": p8,
                "band_16_minus_8": p16 - p8,
                "band_32_minus_16": latent - p16,
                "band_32_minus_8": latent - p8,
            }
            for name, tensor in bands.items():
                sums[name] += tensor.float().square().sum().item()
                per_image_energy[name].append(
                    (
                        tensor.float().square().flatten(1).mean(1)
                        / active_dof_fraction[name]
                    ).cpu().numpy()
                )
            elements += latent.numel()
            images_seen += latent.shape[0]
            if images_seen % 1000 == 0 or images_seen == args.num_images:
                print(f"encoded {images_seen}/{args.num_images}")

    if images_seen == 0:
        raise RuntimeError("dataset produced no images")
    pixel_variances = {name: value / elements for name, value in sums.items()}
    variances = {
        name: value / active_dof_fraction[name]
        for name, value in pixel_variances.items()
    }
    per_image_energy = {
        name: np.concatenate(chunks, axis=0)
        for name, chunks in per_image_energy.items()
    }
    betas = get_named_beta_schedule("linear", 1000)
    alpha_bar = np.cumprod(1.0 - betas)
    raw_fine = first_snr_below(
        alpha_bar, variances["band_32_minus_16"], args.snr_threshold
    )
    raw_middle = first_snr_below(
        alpha_bar, variances["band_16_minus_8"], args.snr_threshold
    )
    raw_direct = first_snr_below(
        alpha_bar, variances["band_32_minus_8"], args.snr_threshold
    )

    band_to_output = {
        "band_32_minus_16": "32_minus_16",
        "band_16_minus_8": "16_minus_8",
        "band_32_minus_8": "32_minus_8",
    }
    crossing_samples = {name: [] for name in band_to_output.values()}
    for band_name, output_name in band_to_output.items():
        for threshold in thresholds:
            crossing_samples[output_name].append(first_snr_below(
                alpha_bar, variances[band_name], threshold
            ))
    rng = np.random.default_rng(args.bootstrap_seed)
    for _ in range(args.bootstrap_replicates):
        indices = rng.integers(0, images_seen, size=images_seen)
        for band_name, output_name in band_to_output.items():
            bootstrap_variance = float(per_image_energy[band_name][indices].mean())
            for threshold in thresholds:
                crossing_samples[output_name].append(first_snr_below(
                    alpha_bar, bootstrap_variance, threshold
                ))
    uncertainty = {
        name: interval_summary(values, args.grid)
        for name, values in crossing_samples.items()
    }

    # Preserve the actual data-only prediction on the 50-step search grid.
    # A very short middle stage is a scientific result, not something the
    # predictor should silently overwrite.  We additionally expose a
    # conservative center with >=100 steps/stage for a robustness control.
    b1_raw_grid = round_boundary(raw_fine, args.grid)
    b2_raw_grid = round_boundary(raw_middle, args.grid)
    predictor_order_valid = args.grid <= b1_raw_grid < b2_raw_grid <= 1000 - args.grid
    b1_raw = min(1000 - 2 * args.grid, max(args.grid, b1_raw_grid))
    b2_raw = min(1000 - args.grid, max(b1_raw + args.grid, b2_raw_grid))
    predicted_three_raw = [b1_raw, b2_raw - b1_raw, 1000 - b2_raw]
    b1 = min(800, max(100, b1_raw_grid))
    b2 = min(900, max(b1 + 100, b2_raw_grid))
    direct = min(900, max(100, round_boundary(raw_direct, args.grid)))
    predicted_three_conservative = [b1, b2 - b1, 1000 - b2]
    predicted_two = [direct, 1000 - direct]
    fine_uncertainty = uncertainty["32_minus_16"]
    coarse_uncertainty = uncertainty["16_minus_8"]
    fine_width = fine_uncertainty["q95"] - fine_uncertainty["q05"]
    coarse_width = coarse_uncertainty["q95"] - coarse_uncertainty["q05"]
    if fine_width == coarse_width:
        uncertain_boundary = "both"
        recommended_delta = max(
            fine_uncertainty["search_radius"], coarse_uncertainty["search_radius"]
        )
    elif fine_width > coarse_width:
        uncertain_boundary = "fine"
        recommended_delta = fine_uncertainty["search_radius"]
    else:
        uncertain_boundary = "coarse"
        recommended_delta = coarse_uncertainty["search_radius"]

    payload = {
        "status": "data_only_prior_not_fid_optimum",
        "data_path": str(args.data_path),
        "vae_path": str(args.vae_path),
        "images": images_seen,
        "posterior": args.posterior,
        "snr_threshold": args.snr_threshold,
        "threshold_sensitivity": thresholds,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_seed": args.bootstrap_seed,
        "grid": args.grid,
        "active_dof_fraction": active_dof_fraction,
        "projection_residual_pixel_mean_square": pixel_variances,
        "projection_residual_active_coefficient_mean_square": variances,
        "raw_first_timestep_with_snr_below_threshold": {
            "32_minus_16": raw_fine,
            "16_minus_8": raw_middle,
            "32_minus_8": raw_direct,
        },
        "boundary_uncertainty_grid": uncertainty,
        "predictor_order_valid_before_legal_projection": predictor_order_valid,
        "raw_grid_boundaries_three_stage": [b1_raw_grid, b2_raw_grid],
        "legal_projection_applied": not predictor_order_valid,
        "recommended_uncertain_boundary": uncertain_boundary,
        "recommended_delta": recommended_delta,
        "predicted_three_stage_finest_to_coarsest": predicted_three_raw,
        "conservative_three_stage_minimum_100": predicted_three_conservative,
        "predicted_two_stage_finest_to_coarsest": predicted_two,
        "next_command": (
            "python 2d/exp/build_rdd_boundary_plan.py "
            f"--pred-three {','.join(map(str, predicted_three_raw))} "
            f"--pred-two {','.join(map(str, predicted_two))} "
            f"--uncertain-boundary {uncertain_boundary} "
            f"--delta {recommended_delta} "
            "--output eval_results/rdd_boundary_plan.json"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
