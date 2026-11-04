"""Summarize latent statistics for the FastWan faux-RDD root-cause study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def scalar(value: torch.Tensor) -> float:
    return float(value.detach().cpu())


def latent_stats(latent: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    value = latent.float()
    ref = reference.float()
    flat = value.flatten()
    ref_flat = ref.flatten()
    delta_t = value[:, :, 1:] - value[:, :, :-1]
    delta_h = value[:, :, :, 1:] - value[:, :, :, :-1]
    delta_w = value[:, :, :, :, 1:] - value[:, :, :, :, :-1]
    rms = torch.sqrt(torch.mean(value.square()))
    ref_rms = torch.sqrt(torch.mean(ref.square()))
    return {
        "mean": scalar(value.mean()),
        "std": scalar(value.std()),
        "rms": scalar(rms),
        "rms_over_fixed": scalar(rms / ref_rms),
        "min": scalar(value.min()),
        "max": scalar(value.max()),
        "finite_fraction": scalar(torch.isfinite(value).float().mean()),
        "cosine_to_fixed_same_seed": scalar(
            torch.nn.functional.cosine_similarity(flat[None], ref_flat[None])[0]
        ),
        "relative_l2_to_fixed_same_seed": scalar(
            torch.linalg.vector_norm(flat - ref_flat) / torch.linalg.vector_norm(ref_flat)
        ),
        "temporal_delta_rms": scalar(torch.sqrt(torch.mean(delta_t.square()))),
        "height_delta_rms": scalar(torch.sqrt(torch.mean(delta_h.square()))),
        "width_delta_rms": scalar(torch.sqrt(torch.mean(delta_w.square()))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-dir", type=Path, required=True)
    parser.add_argument("--fixed-latent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = torch.load(args.fixed_latent, map_location="cpu", weights_only=True)
    summary: dict[str, object] = {
        "reference": str(args.fixed_latent),
        "warning": (
            "Cosine/L2 compare different generated endpoints and are diagnostics, "
            "not perceptual quality metrics."
        ),
        "cases": {},
    }
    for metrics_path in sorted(args.study_dir.glob("*.metrics.json")):
        name = metrics_path.name.removesuffix(".metrics.json")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        latent_path = args.study_dir / f"{name}.latent.pt"
        latent = torch.load(latent_path, map_location="cpu", weights_only=True)
        summary["cases"][name] = {
            "factors": metrics["stage_factors_spatial_temporal"],
            "transition_mode": metrics["transition_mode"],
            "VSA_sparsity": metrics["VSA_sparsity"],
            "denoise_seconds": metrics["denoise_and_boundary_seconds"],
            **latent_stats(latent, reference),
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

