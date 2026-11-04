"""Print a compact table for a paired CSBR benchmark result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("metrics", type=Path)
    args = parser.parse_args()
    data = json.loads(args.metrics.read_text(encoding="utf-8"))
    comparisons = data.get("latent_comparisons", {})

    print("name\tseconds\tspeedup\ttotal_speedup\tcosine\trel_rmse")
    for name, row in data["results"].items():
        comparison = comparisons.get(f"{name}_vs_{data['baseline_name']}", {})
        print(
            f"{name}\t"
            f"{row['denoise_and_boundary_seconds']['median']:.6f}\t"
            f"{row['denoise_speedup_vs_baseline']:.4f}\t"
            f"{row.get('sample_plus_vae_speedup_vs_baseline', float('nan')):.4f}\t"
            f"{comparison.get('cosine', float('nan')):.6f}\t"
            f"{comparison.get('relative_rmse', float('nan')):.6f}"
        )


if __name__ == "__main__":
    main()
