#!/usr/bin/env python3
"""Aggregate repeated RDD training-free attention sweeps.

The script is intentionally dependency-free so it can run both on surfi2 and
on the local Windows workspace while results are arriving incrementally.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


SWEEPS = {
    "centroid_oracle": "rdd_training_free_centroid_sweep",
    "true_kernel": "rdd_training_free_kernel_sweep",
    "cfg_probability_share": "rdd_cfg_probability_share_sweep",
    "adaptive_probability": "rdd_adaptive_probability_sweep",
    "centroid_tuning16": "rdd_centroid_tuning16",
    "probability_tuning16": "rdd_probability_tuning16",
}


def finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-root", type=Path, default=Path("eval_results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/rdd_training_free_attention_summary.json"),
    )
    args = parser.parse_args()

    records: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    source_files: list[str] = []
    for family, dirname in SWEEPS.items():
        directory = args.eval_root / dirname
        for metric_file in sorted(directory.glob("seed_*/metrics.json")):
            data = json.loads(metric_file.read_text(encoding="utf-8"))
            source_files.append(str(metric_file))
            dense = data.get("variants", {}).get("dense", {})
            dense_seconds = finite(dense.get("denoiser_seconds"))
            for name, variant in data.get("variants", {}).items():
                if name == "dense":
                    continue
                seconds = finite(variant.get("denoiser_seconds"))
                speedup = (
                    dense_seconds / seconds
                    if dense_seconds is not None and seconds not in (None, 0.0)
                    else None
                )
                image = variant.get("image_vs_dense", {})
                latent = variant.get("latent_vs_dense", {})
                records[(family, name)].append(
                    {
                        "seed": data.get("seed"),
                        "classes": data.get("classes"),
                        "sample_steps": data.get("sample_steps_coarse_to_fine"),
                        "speedup_vs_dense": speedup,
                        "denoiser_seconds": seconds,
                        "image_cosine": finite(image.get("pixel_cosine")),
                        "image_psnr_db": finite(image.get("psnr_db")),
                        "image_mse": finite(image.get("mse")),
                        "latent_cosine": finite(latent.get("pixel_cosine")),
                        "setting": variant.get("setting"),
                        "stats": {
                            key: value
                            for key, value in variant.items()
                            if key
                            not in {
                                "setting",
                                "image_vs_dense",
                                "latent_vs_dense",
                                "denoiser_seconds",
                                "denoiser_seconds_runs",
                                "denoiser_seconds_mean",
                                "stage_seconds",
                                "stage_seconds_runs",
                            }
                        },
                        "source": str(metric_file),
                    }
                )

    aggregated = []
    for (family, name), runs in sorted(records.items()):
        def values(key: str) -> list[float]:
            return [v for run in runs if (v := finite(run.get(key))) is not None]

        speedups = values("speedup_vs_dense")
        cosines = values("image_cosine")
        psnrs = values("image_psnr_db")
        latent_cosines = values("latent_cosine")
        aggregated.append(
            {
                "family": family,
                "name": name,
                "runs": len(runs),
                "speedup_median": median(speedups),
                "speedup_mean": mean(speedups),
                "image_cosine_mean": mean(cosines),
                "image_cosine_min": min(cosines) if cosines else None,
                "image_psnr_db_mean": mean(psnrs),
                "latent_cosine_mean": mean(latent_cosines),
                "setting": runs[0].get("setting"),
                "per_run": runs,
            }
        )

    # A candidate is Pareto-dominated only within its own timing semantics.
    for row in aggregated:
        row["pareto"] = True
        if row["family"] in {"centroid_oracle", "centroid_tuning16"}:
            # This branch materializes dense logits and cannot claim speed.
            row["speedup_median"] = None
            row["speedup_mean"] = None
            continue
        speed = row.get("speedup_median")
        quality = row.get("image_cosine_mean")
        if speed is None or quality is None:
            row["pareto"] = False
            continue
        for other in aggregated:
            if other is row or other["family"] != row["family"]:
                continue
            other_speed = other.get("speedup_median")
            other_quality = other.get("image_cosine_mean")
            if other_speed is None or other_quality is None:
                continue
            if (
                other_speed >= speed
                and other_quality >= quality
                and (other_speed > speed or other_quality > quality)
            ):
                row["pareto"] = False
                break

    output = {
        "source_files": source_files,
        "complete_seed_counts": {
            family: len(list((args.eval_root / dirname).glob("seed_*/metrics.json")))
            for family, dirname in SWEEPS.items()
        },
        "variants": aggregated,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
