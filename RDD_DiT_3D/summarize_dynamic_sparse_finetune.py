#!/usr/bin/env python3
"""Summarize checkpoint-wise generated support and sparse token cost."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", default=None)
    args = parser.parse_args()

    rows = []
    if args.seed is None:
        pattern = "step*_generated_support.json"
    else:
        pattern = f"step*_seed{args.seed}_generated_support.json"
    for metric_path in sorted(args.eval_dir.glob(pattern)):
        metric_tag = metric_path.name.removesuffix("_generated_support.json")
        checkpoint_tag = metric_tag.split("_seed", maxsplit=1)[0]
        sample_tag = checkpoint_tag
        if args.seed is not None:
            sample_tag = f"{checkpoint_tag}_seed{args.seed}"
        manifest_path = args.run_dir / f"samples_{sample_tag}" / "manifest.json"
        if not manifest_path.exists():
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        steps = manifest.get("steps", [8, 12, 20])
        multiplier = 2 if manifest.get("solver") == "heun" else 1
        token_nfe = [
            sum(
                stage["candidate_tokens"] * step * multiplier
                for stage, step in zip(sample["stages"], steps)
            )
            for sample in manifest["samples"]
        ]
        dense_nfe = sum(
            resolution**3 * step * multiplier
            for resolution, step in zip((8, 16, 32), steps)
        )
        row = {
            "checkpoint": checkpoint_tag,
            "seed": args.seed,
            **metrics["aggregate"],
            "stages": metrics.get("stages", {}),
            "candidate_token_nfe": float(np.mean(token_nfe)),
            "dense_matched_reduction": dense_nfe / max(float(np.mean(token_nfe)), 1.0),
        }
        rows.append(row)

    report = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        stage_recalls = "/".join(
            f"{row['stages'].get(str(resolution), {}).get('support_recall', float('nan')):.3f}"
            for resolution in (8, 16, 32)
        )
        print(
            f"{row['checkpoint']} P={row['support_precision']:.3f} "
            f"R={row['support_recall']:.3f} IoU={row['support_iou']:.3f} "
            f"stageR={stage_recalls} NFE={row['candidate_token_nfe']:.0f} "
            f"reduction={row['dense_matched_reduction']:.2f}x"
        )


if __name__ == "__main__":
    main()
