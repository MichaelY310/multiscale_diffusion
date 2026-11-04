#!/usr/bin/env python3
"""Aggregate paired CSBR layerwise predictor runs across prompts and seeds."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("case_*/paired_suite_metrics.json"))
    if not paths:
        raise RuntimeError(f"no completed cases under {args.root}")
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    pixel_paths = [
        path.parent / "same_seed_pixel_metrics.json" for path in paths
    ]
    if not all(path.is_file() for path in pixel_paths):
        raise RuntimeError("same_seed_pixel_metrics.json is missing in one or more cases")
    pixel_payloads = [
        json.loads(path.read_text(encoding="utf-8")) for path in pixel_paths
    ]
    names = list(payloads[0]["results"])
    rows = []
    for name in names:
        denoise = [
            float(payload["results"][name]["denoise_and_boundary_seconds"]["median"])
            for payload in payloads
        ]
        stage2 = [
            float(payload["results"][name]["stage_forward_seconds"][2]["median"])
            for payload in payloads
        ]
        total = [
            float(payload["results"][name]["sample_plus_vae_seconds"])
            for payload in payloads
            if "sample_plus_vae_seconds" in payload["results"][name]
        ]
        baseline = [
            float(
                payload["results"][payload["baseline_name"]][
                    "denoise_and_boundary_seconds"
                ]["median"]
            )
            for payload in payloads
        ]
        baseline_total = [
            float(
                payload["results"][payload["baseline_name"]][
                    "sample_plus_vae_seconds"
                ]
            )
            for payload in payloads
        ]
        comparison_key = f"{name}_vs_{payloads[0]['baseline_name']}"
        if name == payloads[0]["baseline_name"]:
            latent_cosines = [1.0] * len(payloads)
            latent_rrmses = [0.0] * len(payloads)
        else:
            latent_cosines = [
                float(payload["latent_comparisons"][comparison_key]["cosine"])
                for payload in payloads
            ]
            latent_rrmses = [
                float(payload["latent_comparisons"][comparison_key]["relative_rmse"])
                for payload in payloads
            ]
        has_pixel = all(name in payload for payload in pixel_payloads)
        pixel_ssim = (
            [float(payload[name]["ssim_all"]) for payload in pixel_payloads]
            if has_pixel
            else []
        )
        pixel_psnr = (
            [float(payload[name]["psnr_average_db"]) for payload in pixel_payloads]
            if has_pixel
            else []
        )
        rows.append(
            {
                "name": name,
                "cases": len(payloads),
                "denoise_seconds_mean_of_medians": mean(denoise),
                "denoise_speedup_ratio_of_means": mean(baseline) / mean(denoise),
                "denoise_latency_reduction_percent": (
                    1.0 - mean(denoise) / mean(baseline)
                )
                * 100.0,
                "fine_stage_seconds_mean_of_medians": mean(stage2),
                "sample_plus_vae_seconds_mean": mean(total) if total else None,
                "sample_plus_vae_speedup_ratio_of_means": (
                    mean(baseline_total) / mean(total) if total else None
                ),
                "latent_cosine_mean": mean(latent_cosines),
                "latent_relative_rmse_mean": mean(latent_rrmses),
                "latent_cosine_min": min(latent_cosines),
                "latent_relative_rmse_max": max(latent_rrmses),
                "same_seed_ssim_mean": mean(pixel_ssim) if pixel_ssim else None,
                "same_seed_psnr_db_mean": mean(pixel_psnr) if pixel_psnr else None,
            }
        )
    output = {
        "protocol": {
            "cases": len(payloads),
            "aggregation": "mean of per-case medians; ratio of aggregate means",
            "pixel_metrics": (
                "same prompt/seed perturbation from RDD-VSA20 baseline; "
                "not a perceptual or generative-quality metric"
            ),
        },
        "rows": rows,
    }
    json_path = args.root / "aggregate_metrics.json"
    csv_path = args.root / "aggregate_metrics.csv"
    json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
