#!/usr/bin/env python3
"""Merge formal held-out support, compute, and mesh metrics into JSON/Markdown."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np


def mean_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "std": float(array.std(ddof=0))}


def fmt(summary: dict[str, float] | None, digits: int = 3) -> str:
    if not summary:
        return "--"
    return f"{summary['mean']:.{digits}f}±{summary['std']:.{digits}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()

    seed_rows = []
    pattern = re.compile(r"(?P<method>.+)_seed(?P<seed>\d+)_heldout_support\.json$")
    for metric_path in sorted(args.eval_root.glob("*_seed*_heldout_support.json")):
        match = pattern.match(metric_path.name)
        if match is None:
            continue
        method = match.group("method")
        seed = int(match.group("seed"))
        manifest_path = args.output_root / f"samples_{method}_seed{seed}" / "manifest.json"
        if not manifest_path.is_file():
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        steps = [int(value) for value in manifest.get("steps", (8, 12, 20))]
        resolutions = (8, 16, 32)
        multiplier = 2 if manifest.get("solver") == "heun" else 1
        per_sample_nfe = [
            sum(
                int(stage["candidate_tokens"]) * step * multiplier
                for stage, step in zip(sample["stages"], steps)
            )
            for sample in manifest["samples"]
        ]
        dense_nfe = sum(
            resolution**3 * step * multiplier
            for resolution, step in zip(resolutions, steps)
        )
        candidate_nfe = float(np.mean(per_sample_nfe))
        wall_seconds = [
            float(sample["sampling_wall_seconds"])
            for sample in manifest["samples"]
            if sample.get("sampling_wall_seconds") is not None
        ]
        peak_bytes = [
            int(sample["cuda_peak_allocated_bytes"])
            for sample in manifest["samples"]
            if sample.get("cuda_peak_allocated_bytes") is not None
        ]
        row = {
            "method": method,
            "seed": seed,
            **metrics["aggregate"],
            "stage_recall": {
                str(resolution): metrics.get("stages", {})
                .get(str(resolution), {})
                .get("support_recall")
                for resolution in resolutions
            },
            "candidate_token_nfe": candidate_nfe,
            "matched_dense_token_nfe": dense_nfe,
            "dense_matched_reduction": dense_nfe / max(candidate_nfe, 1.0),
        }
        if wall_seconds:
            row["sampling_wall_seconds_per_asset"] = float(np.mean(wall_seconds))
        if peak_bytes:
            row["cuda_peak_allocated_mib"] = float(max(peak_bytes) / 2**20)
        mesh_path = args.eval_root / f"{method}_seed{seed}_mesh.json"
        if mesh_path.is_file():
            mesh = json.loads(mesh_path.read_text(encoding="utf-8"))
            row["mesh"] = mesh.get("aggregate", {})
            row["mesh_failures"] = len(mesh.get("failures", []))
        seed_rows.append(row)

    if not seed_rows:
        raise SystemExit("no formal held-out support reports with matching manifests")

    methods = []
    for method in dict.fromkeys(row["method"] for row in seed_rows):
        rows = [row for row in seed_rows if row["method"] == method]
        aggregate = {"method": method, "seeds": [row["seed"] for row in rows]}
        for key in (
            "support_precision",
            "support_recall",
            "support_iou",
            "feature_mse_on_intersection",
            "candidate_token_nfe",
            "dense_matched_reduction",
            "sampling_wall_seconds_per_asset",
            "cuda_peak_allocated_mib",
        ):
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            if values:
                aggregate[key] = mean_std(values)
        aggregate["stage_recall"] = {}
        for resolution in (8, 16, 32):
            values = [
                float(row["stage_recall"][str(resolution)])
                for row in rows
                if row["stage_recall"].get(str(resolution)) is not None
            ]
            if values:
                aggregate["stage_recall"][str(resolution)] = mean_std(values)
        for key in (
            "chamfer_l2_squared_mean",
            "precision_mean",
            "recall_mean",
            "fscore_mean",
        ):
            values = [
                float(row["mesh"][key])
                for row in rows
                if key in row.get("mesh", {})
            ]
            if values:
                aggregate.setdefault("mesh", {})[key] = mean_std(values)
        methods.append(aggregate)

    report = {"methods": methods, "per_seed": seed_rows}
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "| Method | Support P | Support R | IoU | Stage recall 8/16/32 | token-NFE | Dense reduction | Wall/asset (s) | Peak MiB | Chamfer² | Mesh F-score |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in methods:
        stage = "/".join(
            fmt(row.get("stage_recall", {}).get(str(resolution)))
            for resolution in (8, 16, 32)
        )
        mesh = row.get("mesh", {})
        lines.append(
            "| {method} | {precision} | {recall} | {iou} | {stage} | "
            "{nfe} | {reduction} | {wall} | {peak} | {chamfer} | {fscore} |".format(
                method=row["method"],
                precision=fmt(row.get("support_precision")),
                recall=fmt(row.get("support_recall")),
                iou=fmt(row.get("support_iou")),
                stage=stage,
                nfe=fmt(row.get("candidate_token_nfe"), digits=0),
                reduction=fmt(row.get("dense_matched_reduction"), digits=2),
                wall=fmt(row.get("sampling_wall_seconds_per_asset"), digits=3),
                peak=fmt(row.get("cuda_peak_allocated_mib"), digits=0),
                chamfer=fmt(mesh.get("chamfer_l2_squared_mean"), digits=6),
                fscore=fmt(mesh.get("fscore_mean")),
            )
        )
    args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output_markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
