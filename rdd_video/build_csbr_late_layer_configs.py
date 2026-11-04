#!/usr/bin/env python3
"""Build end-to-end sensitivity guided CSBR layer-set candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictor-analysis", type=Path, required=True)
    parser.add_argument("--sensitivity-ranking", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predictor = json.loads(args.predictor_analysis.read_text(encoding="utf-8"))
    ranking = json.loads(args.sensitivity_ranking.read_text(encoding="utf-8"))
    coefficients = {
        int(row["layer"]): row for row in predictor["fit_all_coefficients"]
    }
    ranked_layers = [int(row["layer"]) for row in ranking]

    def candidate(name: str, layers: list[int], use_trend: bool) -> dict:
        layer_coefficients = {}
        for layer in sorted(layers):
            row = coefficients[layer]
            if use_trend:
                values = [float(row["a"]), float(row["b"]), 0.0]
            else:
                values = [float(row["scalar_middle"]), 0.0, 0.0]
            layer_coefficients[str(layer)] = values
        return {
            "name": name,
            "stage_factors": "4,1;2,1;1,1",
            "attention_mode": "vsa",
            "boundary_executor": "torch_nearest",
            "cross_stage_reuse_layers": sorted(layers),
            "cross_stage_reuse_stages": [2],
            "cross_stage_reuse_layer_coefficients": layer_coefficients,
        }

    top4 = ranked_layers[:4]
    top6 = ranked_layers[:6]
    top8 = ranked_layers[:8]
    configs = [
        {
            "name": "rdd_vsa20_baseline",
            "stage_factors": "4,1;2,1;1,1",
            "attention_mode": "vsa",
            "boundary_executor": "torch_nearest",
        },
        {
            "name": "uniform6_global075",
            "stage_factors": "4,1;2,1;1,1",
            "attention_mode": "vsa",
            "boundary_executor": "torch_nearest",
            "cross_stage_reuse_layers": [0, 5, 10, 15, 20, 25],
            "cross_stage_reuse_stages": [2],
            "cross_stage_reuse_scale": 0.75,
        },
        candidate("safe4_multiscale", top4, True),
        candidate("safe6_scalar", top6, False),
        candidate("safe6_multiscale", top6, True),
        candidate("tail6_multiscale", [24, 25, 26, 27, 28, 29], True),
        candidate("safe8_multiscale", top8, True),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    print("ranked layers:", ranked_layers)
    print(f"wrote {len(configs)} configs to {args.output}")


if __name__ == "__main__":
    main()
