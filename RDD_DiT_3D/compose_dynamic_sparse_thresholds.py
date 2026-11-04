#!/usr/bin/env python3
"""Compose stage thresholds while preserving a closed-loop calibration trail."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


def compose_thresholds(
    base: dict,
    generated: dict,
    replace_resolutions: tuple[int, ...],
) -> dict:
    replacement = {
        int(stage["resolution"]): stage for stage in generated.get("stages", ())
    }
    requested = set(int(value) for value in replace_resolutions)
    missing = requested - replacement.keys()
    if missing:
        raise ValueError(f"generated report lacks resolutions: {sorted(missing)}")
    output = copy.deepcopy(base)
    stages = output.get("stages")
    if not isinstance(stages, list):
        raise ValueError("base report must contain a stages list")
    seen = set()
    for stage in stages:
        resolution = int(stage["resolution"])
        if resolution not in requested:
            continue
        source = replacement[resolution]
        stage["probability_threshold"] = float(source["probability_threshold"])
        stage["closed_loop_source"] = {
            key: source.get(key)
            for key in (
                "measured_recall",
                "measured_precision",
                "selection_score",
                "candidate_coverage_of_gt",
                "global_recall_at_threshold",
                "tokens",
                "positives_in_candidates",
                "ground_truth_positives",
            )
            if key in source
        }
        seen.add(resolution)
    missing_base = requested - seen
    if missing_base:
        raise ValueError(f"base report lacks resolutions: {sorted(missing_base)}")
    history = list(output.get("closed_loop_history", ()))
    history.append(
        {
            "generated_report": generated.get("source_path"),
            "objective": generated.get("objective"),
            "fbeta": generated.get("fbeta"),
            "replaced_resolutions": sorted(requested),
            "replay_dirs": generated.get("replay_dirs"),
        }
    )
    output["closed_loop_history"] = history
    output["note"] = (
        "sequential closed-loop generated-state policy; each refinement threshold "
        "was calibrated only after rolling out all preceding calibrated stages"
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--replace-resolutions", type=int, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base.read_text(encoding="utf-8"))
    generated = json.loads(args.generated.read_text(encoding="utf-8"))
    generated["source_path"] = str(args.generated)
    output = compose_thresholds(base, generated, tuple(args.replace_resolutions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
