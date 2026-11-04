#!/usr/bin/env python3
"""Summarize metrics and latent deviations for the faux-RDD Wan study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def similarity(candidate: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    x = candidate.float().flatten()
    y = reference.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(x, y, dim=0)
    relative_l2 = (x - y).norm() / y.norm().clamp_min(1e-12)
    return {"latent_cosine": float(cosine), "latent_relative_l2": float(relative_l2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.input_dir)
    metrics = {}
    for path in sorted(root.glob("*.metrics.json")):
        metrics[path.name.removesuffix(".metrics.json")] = json.loads(path.read_text())

    references = {}
    for name in ("fullres_dense", "boundary_only_dense", "faux_rdd_dense"):
        path = root / f"{name}.latent.pt"
        if path.exists():
            references[name] = torch.load(path, map_location="cpu", weights_only=True)

    rows = []
    for name, item in metrics.items():
        row = {
            "name": name,
            "denoise_seconds": item.get("denoise_and_boundary_seconds"),
            "end_to_end_seconds": item.get("end_to_end_inference_seconds"),
            "peak_cuda_gib": item.get("peak_cuda_memory_bytes", 0) / 2**30,
            "attention": item.get("attention"),
        }
        latent_path = root / f"{name}.latent.pt"
        if latent_path.exists():
            latent = torch.load(latent_path, map_location="cpu", weights_only=True)
            for ref_name, reference in references.items():
                if latent.shape == reference.shape:
                    row[f"vs_{ref_name}"] = similarity(latent, reference)
        rows.append(row)
    payload = {
        "warning": (
            "Original Wan weights were not trained for the RDD drift or sparse route. "
            "These results validate code paths and measure additional degradation only."
        ),
        "rows": rows,
    }
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

