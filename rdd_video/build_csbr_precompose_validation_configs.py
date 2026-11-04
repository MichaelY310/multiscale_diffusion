#!/usr/bin/env python3
"""Build a paired suite that isolates middle-stage predictor precomposition."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


KEEP = {
    "rdd_vsa20_baseline",
    "safe6_scalar",
    "safe6_multiscale",
    "safe8_multiscale",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="rdd_video/experiments/fastwan_csbr_late_layer_candidates.json",
    )
    parser.add_argument(
        "--output",
        default="rdd_video/experiments/fastwan_csbr_precompose_validation.json",
    )
    args = parser.parse_args()

    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    selected = [entry for entry in source if entry["name"] in KEEP]
    by_name = {entry["name"]: entry for entry in selected}
    missing = KEEP - set(by_name)
    if missing:
        raise KeyError(f"source config is missing: {sorted(missing)}")

    configs = [
        by_name["rdd_vsa20_baseline"],
        by_name["safe6_scalar"],
        by_name["safe6_multiscale"],
    ]
    safe6_pre = copy.deepcopy(by_name["safe6_multiscale"])
    safe6_pre["name"] = "safe6_multiscale_precompose"
    safe6_pre["cross_stage_reuse_precompose"] = True
    configs.append(safe6_pre)
    configs.append(by_name["safe8_multiscale"])
    safe8_pre = copy.deepcopy(by_name["safe8_multiscale"])
    safe8_pre["name"] = "safe8_multiscale_precompose"
    safe8_pre["cross_stage_reuse_precompose"] = True
    configs.append(safe8_pre)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(configs)} configs to {output}")


if __name__ == "__main__":
    main()
