#!/usr/bin/env python3
"""Build a small successive-halving plan around predicted RDD boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def split(value: str, count: int) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if len(result) != count or any(item <= 0 for item in result) or sum(result) != 1000:
        raise ValueError(f"expected {count} positive lengths summing to 1000, got {value}")
    return result


def valid(candidate: tuple[int, ...], minimum: int = 50) -> bool:
    return sum(candidate) == 1000 and all(value >= minimum for value in candidate)


def add_unique(
    output: list[tuple[int, ...]], candidate: tuple[int, ...],
    excluded: set[tuple[int, ...]], limit: int,
) -> None:
    if len(output) < limit and valid(candidate) and candidate not in excluded and candidate not in output:
        output.append(candidate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-three", default="300,300,400")
    parser.add_argument("--pred-two", default="400,600")
    parser.add_argument(
        "--uncertain-boundary", choices=["fine", "coarse", "both"], default="fine"
    )
    parser.add_argument("--delta", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    predicted_three = split(args.pred_three, 3)
    predicted_two = split(args.pred_two, 2)
    existing = {(100, 300, 600), (300, 300, 400)}
    delta = args.delta
    if delta < 50 or delta > 150 or delta % 50:
        raise ValueError("--delta must be a 50-step grid value in [50, 150]")

    l32, l16, l8 = predicted_three
    neighbours = []
    if args.uncertain_boundary in {"fine", "both"}:
        neighbours.extend([
            (l32 - delta, l16 + delta, l8),
            (l32 + delta, l16 - delta, l8),
        ])
    if args.uncertain_boundary in {"coarse", "both"}:
        neighbours.extend([
            (l32, l16 - delta, l8 + delta),
            (l32, l16 + delta, l8 - delta),
        ])
    pool_three = [
        predicted_three,
        *neighbours,
    ]
    new_three: list[tuple[int, ...]] = []
    for candidate in pool_three:
        add_unique(new_three, candidate, existing, 3)
    # When a predicted/local point exactly matches a reusable checkpoint, do
    # not invent a remote stress-test merely to keep the new-run count at 3.
    # Existing exact checkpoints remain part of the observation table.

    fine_two, coarse_two = predicted_two
    pool_two = [
        predicted_two,
        (fine_two - delta, coarse_two + delta),
        (fine_two + delta, coarse_two - delta),
        (500, 500),
    ]
    new_two: list[tuple[int, ...]] = []
    for candidate in pool_two:
        if valid(candidate) and candidate not in new_two:
            new_two.append(candidate)
    if not new_two:
        raise RuntimeError("no legal 2-stage candidates")

    rows = []
    for topology, candidates in [((32, 16, 8), new_three), ((32, 8), new_two)]:
        for lengths in candidates:
            tag = "-".join(str(value) for value in lengths)
            rows.append({
                "topology": list(topology),
                "stage_lengths": list(lengths),
                "name": f"RDD-boundary-{'-'.join(map(str, topology))}-L{tag}",
                "phase_1_max_steps": 50_000,
                "phase_2_max_steps": 100_000,
                "final_max_steps": 400_000,
                "phase_1_samples": 2_000,
                "phase_2_samples": 10_000,
            })

    payload = {
        "existing_reusable_three_stage": [list(item) for item in sorted(existing)],
        "predicted_three_stage": list(predicted_three),
        "predicted_two_stage": list(predicted_two),
        "uncertain_boundary": args.uncertain_boundary,
        "delta": delta,
        "new_candidates": rows,
        "decision": {
            "after_50k": "only eliminate clearly broken candidates",
            "after_100k": "continue top 2 per topology to 400k",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    print("\nPhase-1 commands (run each in its own tmux/GPU allocation):")
    for row in rows:
        resolutions = ",".join(map(str, row["topology"]))
        lengths = ",".join(map(str, row["stage_lengths"]))
        print(
            f"STAGE_RESOLUTIONS={resolutions} STAGE_LENGTHS={lengths} "
            f"RESULTS_NAME={row['name']} MAX_TRAIN_STEPS=50000 "
            "bash 2d/exp/train_rdd_boundary_candidate.sh"
        )


if __name__ == "__main__":
    main()
