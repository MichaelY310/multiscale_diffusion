#!/usr/bin/env python3
"""Build one-skipped-layer CSBR configs from fitted multi-scale coefficients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.analysis.read_text(encoding="utf-8"))
    configs = [
        {
            "name": "rdd_vsa20_baseline",
            "stage_factors": "4,1;2,1;1,1",
            "attention_mode": "vsa",
            "boundary_executor": "torch_nearest",
        }
    ]
    for row in payload["fit_all_coefficients"]:
        layer = int(row["layer"])
        configs.append(
            {
                "name": f"single_l{layer:02d}_multiscale",
                "stage_factors": "4,1;2,1;1,1",
                "attention_mode": "vsa",
                "boundary_executor": "torch_nearest",
                "cross_stage_reuse_layers": [layer],
                "cross_stage_reuse_stages": [2],
                "cross_stage_reuse_layer_coefficients": {
                    str(layer): [float(row["a"]), float(row["b"]), 0.0]
                },
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(configs, indent=2), encoding="utf-8")
    print(f"wrote {len(configs)} configs to {args.output}")


if __name__ == "__main__":
    main()
