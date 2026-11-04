#!/usr/bin/env python3
"""Build a compact 1280x768 validation suite for sensitivity-guided CSBR."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--late-source",
        default="rdd_video/experiments/fastwan_csbr_late_layer_candidates.json",
    )
    parser.add_argument(
        "--stage-source",
        default="rdd_video/experiments/fastwan_csbr_stage_specific_validation.json",
    )
    parser.add_argument(
        "--output",
        default="rdd_video/experiments/fastwan_csbr_highres_validation.json",
    )
    args = parser.parse_args()
    late = {
        row["name"]: row
        for row in json.loads(Path(args.late_source).read_text(encoding="utf-8"))
    }
    stage = {
        row["name"]: row
        for row in json.loads(Path(args.stage_source).read_text(encoding="utf-8"))
    }
    configs = [
        late["rdd_vsa20_baseline"],
        late["uniform6_global075"],
        stage["fine_safe6"],
        stage["combined_disjoint_m2_f6"],
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(configs)} configs to {output}")


if __name__ == "__main__":
    main()
