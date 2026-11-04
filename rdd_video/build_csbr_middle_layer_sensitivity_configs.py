#!/usr/bin/env python3
"""Build one-layer coarse-to-middle CSBR sensitivity configs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "rdd_video/experiments/fastwan_csbr_middle_layer_sensitivity.json"
        ),
    )
    parser.add_argument("--scale", type=float, default=0.75)
    args = parser.parse_args()
    configs = [
        {
            "name": "rdd_vsa20_baseline",
            "stage_factors": "4,1;2,1;1,1",
            "attention_mode": "vsa",
            "boundary_executor": "torch_nearest",
        }
    ]
    for layer in range(30):
        configs.append(
            {
                "name": f"single_s1_l{layer:02d}_scalar075",
                "stage_factors": "4,1;2,1;1,1",
                "attention_mode": "vsa",
                "boundary_executor": "torch_nearest",
                "cross_stage_reuse_layers": [layer],
                "cross_stage_reuse_stages": [1],
                "cross_stage_reuse_scale": args.scale,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(configs)} configs to {args.output}")


if __name__ == "__main__":
    main()
