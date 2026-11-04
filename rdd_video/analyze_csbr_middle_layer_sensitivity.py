#!/usr/bin/env python3
"""Rank coarse-to-middle one-layer CSBR substitutions by final-latent error."""

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
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.root.glob("case_*/paired_suite_metrics.json"))
    ]
    if not payloads:
        raise RuntimeError(f"no completed cases under {args.root}")
    baseline = payloads[0]["baseline_name"]
    rows = []
    for layer in range(30):
        name = f"single_s1_l{layer:02d}_scalar075"
        key = f"{name}_vs_{baseline}"
        cosines = [float(p["latent_comparisons"][key]["cosine"]) for p in payloads]
        rrmses = [
            float(p["latent_comparisons"][key]["relative_rmse"]) for p in payloads
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
    (args.root / "middle_layer_sensitivity_ranking.json").write_text(
        json.dumps(rows, indent=2) + "\n", encoding="utf-8"
    )
    with (args.root / "middle_layer_sensitivity_ranking.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
