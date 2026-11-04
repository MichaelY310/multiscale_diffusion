"""Aggregate the small multi-prompt CSBR residual diagnostic."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


METRICS = {
    "raw_cosine": ("matched_to_raw", "cosine"),
    "raw_relative_rmse": ("matched_to_raw", "relative_rmse"),
    "raw_optimal_relative_rmse": ("matched_to_raw", "optimal_relative_rmse"),
    "stage_low_cosine": ("matched_to_stage_low", "cosine"),
    "stage_low_relative_rmse": ("matched_to_stage_low", "relative_rmse"),
    "stage_low_optimal_relative_rmse": (
        "matched_to_stage_low",
        "optimal_relative_rmse",
    ),
    "stage_low_optimal_scalar": ("matched_to_stage_low", "optimal_scalar"),
    "scalar_gradient_at_one": (
        "matched_to_stage_low",
        "scalar_gradient_at_one",
    ),
    "stage_high_cosine_tautological": ("matched_to_stage_high_cosine",),
    "wrong_layer_stage_low_cosine": ("wrong_layer_to_stage_low_cosine",),
    "matched_minus_wrong_low_cosine": ("matched_minus_wrong_low_cosine",),
    "stage_low_energy_fraction": ("stage_low_energy_fraction",),
    "stage_high_energy_fraction": ("stage_high_energy_fraction",),
    "error_high_to_low_rms_ratio": ("error_high_to_low_rms_ratio",),
    "same_grid_half_low_cosine": ("same_grid_half_low_cosine",),
    "same_grid_half_high_cosine": ("same_grid_half_high_cosine",),
}


def get_path(row: dict[str, Any], path: Iterable[str]) -> float:
    value: Any = row
    for key in path:
        value = value[key]
    return float(value)


def describe(values: list[float]) -> dict[str, float]:
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def main(args: argparse.Namespace) -> None:
    root = Path(args.input_dir).resolve()
    files = sorted(root.glob("case_*/metrics.json"))
    if not files:
        raise FileNotFoundError(f"no case_*/metrics.json under {root}")
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    rows = [row for case in cases for row in case["residual_rows"]]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    layer_grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["run"], row["boundary"])].append(row)
        layer_grouped[(row["run"], row["boundary"], int(row["layer"]))].append(row)

    summaries: dict[str, Any] = {}
    for (run, boundary), group in sorted(grouped.items()):
        key = f"{run}/{boundary}"
        summaries[key] = {
            metric: describe([get_path(row, path) for row in group])
            for metric, path in METRICS.items()
        }

    layer_summaries: dict[str, Any] = {}
    for (run, boundary, layer), group in sorted(layer_grouped.items()):
        key = f"{run}/{boundary}/layer_{layer:02d}"
        layer_summaries[key] = {
            metric: describe([get_path(row, path) for row in group])
            for metric, path in METRICS.items()
        }

    comparison_groups: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for case in cases:
        for name, values in case["latent_comparisons"].items():
            for metric, value in values.items():
                comparison_groups[name][metric].append(float(value))
    comparisons = {
        name: {metric: describe(values) for metric, values in metrics.items()}
        for name, metrics in comparison_groups.items()
    }

    timing_groups: dict[str, list[float]] = defaultdict(list)
    for case in cases:
        for name, result in case["results"].items():
            timing_groups[name].append(float(result["denoise_seconds"]))
    timings = {name: describe(values) for name, values in timing_groups.items()}

    # Across-data variance of the scalar-adapter gradient for each layer.  This
    # is the only honest variance available without a training loss/backward.
    gradient_variance_by_run: dict[str, Any] = {}
    for run in sorted({row["run"] for row in rows}):
        per_layer_variances = []
        for layer in range(30):
            group = layer_grouped.get((run, "stage1_to_stage2", layer), [])
            values = [
                get_path(row, METRICS["scalar_gradient_at_one"]) for row in group
            ]
            if len(values) > 1:
                per_layer_variances.append(statistics.variance(values))
        if per_layer_variances:
            gradient_variance_by_run[run] = describe(per_layer_variances)

    payload = {
        "case_count": len(cases),
        "source_files": [str(path) for path in files],
        "residual_summary": summaries,
        "layer_summary": layer_summaries,
        "latent_comparison_summary": comparisons,
        "timing_summary": timings,
        "scalar_adapter_gradient_variance_across_cases_by_layer": (
            gradient_variance_by_run
        ),
        "caveats": {
            "checkpoint": "FastWan is not RDD/CSBR-finetuned",
            "stage_high_cosine": (
                "The lifted coarse tensor is in the stage-low subspace, so its "
                "cosine with the orthogonal stage-high component is near zero by definition."
            ),
            "gradient": (
                "This is variance of the MSE gradient for one residual scale at s=1, "
                "not full-network gradient variance."
            ),
            "timing": "GPUs were shared; timing is diagnostic only.",
        },
    }
    output = root / "aggregate_metrics.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with (root / "residual_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["run", "boundary", "metric", "mean", "stdev", "min", "max"])
        for key, metrics in summaries.items():
            run, boundary = key.split("/", 1)
            for metric, values in metrics.items():
                writer.writerow(
                    [
                        run,
                        boundary,
                        metric,
                        values["mean"],
                        values["stdev"],
                        values["min"],
                        values["max"],
                    ]
                )

    with (root / "latent_comparisons.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["comparison", "metric", "mean", "stdev", "min", "max"])
        for name, metrics in comparisons.items():
            for metric, values in metrics.items():
                writer.writerow(
                    [
                        name,
                        metric,
                        values["mean"],
                        values["stdev"],
                        values["min"],
                        values["max"],
                    ]
                )

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    main(parser.parse_args())
