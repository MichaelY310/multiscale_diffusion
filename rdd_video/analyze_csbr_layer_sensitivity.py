#!/usr/bin/env python3
"""Rank CSBR layers by actual final-latent perturbation across held-out prompts."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    paths = sorted(args.root.glob("case_*/paired_suite_metrics.json"))
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not payloads:
        raise RuntimeError(f"no completed cases under {args.root}")
    baseline = payloads[0]["baseline_name"]
    rows = []
    for layer in range(30):
        name = f"single_l{layer:02d}_multiscale"
        key = f"{name}_vs_{baseline}"
        cosines = [
            float(payload["latent_comparisons"][key]["cosine"])
            for payload in payloads
        ]
        rrmses = [
            float(payload["latent_comparisons"][key]["relative_rmse"])
            for payload in payloads
        ]
        rows.append(
            {
                "layer": layer,
                "cosine_mean": statistics.mean(cosines),
                "cosine_min": min(cosines),
                "relative_rmse_mean": statistics.mean(rrmses),
                "relative_rmse_max": max(rrmses),
                "cases": len(payloads),
            }
        )
    rows.sort(key=lambda row: (row["relative_rmse_mean"], -row["cosine_mean"]))
    output = args.root / "layer_sensitivity_ranking.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (args.root / "layer_sensitivity_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
