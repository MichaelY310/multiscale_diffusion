#!/usr/bin/env python3
"""Build stage-specific CSBR candidates from the measured safe tail layers."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def coeff_subset(source: dict, layers: list[int]) -> dict[str, list[float]]:
    return {str(layer): source[str(layer)] for layer in layers}


def stage_specific(
    name: str,
    middle_layers: list[int],
    fine_layers: list[int],
    fine_coefficients: dict[str, list[float]],
) -> dict:
    layers_by_stage = {}
    coefficients_by_stage = {}
    if middle_layers:
        layers_by_stage["1"] = middle_layers
        coefficients_by_stage["1"] = {
            str(layer): [0.75, 0.0, 0.0] for layer in middle_layers
        }
    if fine_layers:
        layers_by_stage["2"] = fine_layers
        coefficients_by_stage["2"] = coeff_subset(
            fine_coefficients, fine_layers
        )
    return {
        "name": name,
        "stage_factors": "4,1;2,1;1,1",
        "attention_mode": "vsa",
        "boundary_executor": "torch_nearest",
        "cross_stage_reuse_stages": sorted(int(stage) for stage in layers_by_stage),
        "cross_stage_reuse_layers_by_stage": layers_by_stage,
        "cross_stage_reuse_layer_coefficients_by_stage": coefficients_by_stage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="rdd_video/experiments/fastwan_csbr_late_layer_candidates.json",
    )
    parser.add_argument(
        "--output",
        default="rdd_video/experiments/fastwan_csbr_stage_specific_validation.json",
    )
    args = parser.parse_args()
    source = json.loads(Path(args.source).read_text(encoding="utf-8"))
    by_name = {entry["name"]: entry for entry in source}
    baseline = copy.deepcopy(by_name["rdd_vsa20_baseline"])
    all_coefficients = by_name["safe8_multiscale"][
        "cross_stage_reuse_layer_coefficients"
    ]
    fine_safe6 = [23, 24, 25, 26, 28, 29]
    fine_disjoint6 = [22, 23, 24, 25, 26, 27]
    configs = [
        baseline,
        stage_specific("middle_safe2", [28, 29], [], all_coefficients),
        stage_specific("middle_safe4", [26, 27, 28, 29], [], all_coefficients),
        stage_specific("fine_safe6", [], fine_safe6, all_coefficients),
        stage_specific("fine_disjoint6", [], fine_disjoint6, all_coefficients),
        stage_specific(
            "combined_disjoint_m2_f6",
            [28, 29],
            fine_disjoint6,
            all_coefficients,
        ),
        stage_specific(
            "combined_overlap_m2_f6",
            [28, 29],
            fine_safe6,
            all_coefficients,
        ),
        stage_specific(
            "combined_disjoint_m4_f4",
            [26, 27, 28, 29],
            [22, 23, 24, 25],
            all_coefficients,
        ),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(configs, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(configs)} configs to {output}")


if __name__ == "__main__":
    main()
