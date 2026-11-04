#!/usr/bin/env python3
"""Summarize support quality and actual candidate-token NFE for ablations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for metric_path in sorted(args.eval_root.glob("*.json")):
        sample_dir = args.sample_root / metric_path.stem
        manifest_path = sample_dir / "manifest.json"
        if not manifest_path.exists():
            continue
        metrics = json.loads(metric_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        multiplier = 2 if manifest.get("solver") == "heun" else 1
        steps = manifest.get("steps", [8, 12, 20])
        token_nfe = []
        dense_nfe = sum(
            resolution**3 * step * multiplier
            for resolution, step in zip((8, 16, 32), steps)
        )
        for sample in manifest["samples"]:
            token_nfe.append(
                sum(
                    stage["candidate_tokens"] * step * multiplier
                    for stage, step in zip(sample["stages"], steps)
                )
            )
        row = {
            "name": metric_path.stem,
            "solver": manifest.get("solver", "euler"),
            "steps": steps,
            "force_keep_resolutions": manifest.get("force_keep_resolutions", []),
            "halo_radii": manifest.get("halo_radii", [0, 0, 0]),
            **metrics["aggregate"],
            "candidate_token_nfe": float(np.mean(token_nfe)),
            "dense_matched_nfe": dense_nfe,
            "dense_matched_reduction": dense_nfe / max(float(np.mean(token_nfe)), 1),
            "stages": metrics.get("stages", {}),
        }
        rows.append(row)
    report = {"rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    for row in rows:
        stages = row["stages"]
        recalls = "/".join(
            f"{stages.get(str(resolution), {}).get('support_recall', float('nan')):.3f}"
            for resolution in (8, 16, 32)
        )
        print(
            f"{row['name']:20s} P={row['support_precision']:.3f} "
            f"R={row['support_recall']:.3f} IoU={row['support_iou']:.3f} "
            f"stageR={recalls} tokenNFE={row['candidate_token_nfe']:.0f} "
            f"reduction={row['dense_matched_reduction']:.2f}x"
        )


if __name__ == "__main__":
    main()
