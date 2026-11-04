#!/usr/bin/env python3
"""Create a deterministic, auditable TRELLIS-500K ObjaverseXL subset.

The upstream toolkit uses ``metadata.csv`` as the authoritative instance list.
Writing the subset before running ``build_metadata.py`` prevents an accidental
full-dataset download.  Selection is hash-ranked rather than row-order sampled,
so it is stable across pandas versions and source CSV reordering.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def _read_exclusions(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        if path.is_dir():
            # A previous upstream build_metadata.py may expand metadata.csv to
            # the entire source catalog.  The latent filenames are the
            # authoritative record of assets actually used by an old run.
            excluded.update(item.stem for item in path.rglob("*.npz"))
        elif path.suffix.lower() == ".csv":
            frame = pd.read_csv(path, usecols=["sha256"])
            excluded.update(frame["sha256"].astype(str))
        elif path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            values = payload.get("sha256", payload.get("instances", []))
            excluded.update(map(str, values))
        else:
            excluded.update(
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
    return excluded


def _rank(seed: int, sha256: str) -> str:
    return hashlib.sha256(f"rdd3d:{seed}:{sha256}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source", choices=("sketchfab", "github"), default="sketchfab")
    parser.add_argument("--source-csv", default=None)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--split", choices=("train", "validation", "test"), required=True)
    parser.add_argument("--min-aesthetic-score", type=float, default=5.5)
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.count <= 0:
        raise ValueError("--count must be positive")
    source_csv = args.source_csv or (
        "https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K/"
        f"resolve/main/ObjaverseXL_{args.source}.csv"
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / "metadata.csv"
    manifest_path = args.output_root / "rdd3d_subset_manifest.json"
    if (output.exists() or manifest_path.exists()) and not args.overwrite:
        raise SystemExit(
            f"subset already exists under {args.output_root}; pass --overwrite explicitly"
        )

    frame = pd.read_csv(source_csv)
    required = {"sha256", "file_identifier", "aesthetic_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"source metadata is missing columns: {missing}")
    source_count = len(frame)
    frame = frame.drop_duplicates("sha256")
    frame = frame[frame["aesthetic_score"] >= args.min_aesthetic_score].copy()
    eligible_before_exclusion = len(frame)
    excluded = _read_exclusions(args.exclude)
    if excluded:
        frame = frame[~frame["sha256"].astype(str).isin(excluded)].copy()
    if len(frame) < args.count:
        raise ValueError(
            f"only {len(frame)} eligible objects remain, fewer than requested {args.count}"
        )

    frame["_rdd3d_rank"] = [
        _rank(args.seed, value) for value in frame["sha256"].astype(str)
    ]
    selected = frame.sort_values("_rdd3d_rank", kind="stable").head(args.count).copy()
    selected.drop(columns=["_rdd3d_rank"], inplace=True)
    selected["rdd3d_split"] = args.split
    selected.to_csv(output, index=False)

    ordered_hashes = selected["sha256"].astype(str).tolist()
    manifest = {
        "schema_version": 1,
        "source": args.source,
        "source_csv": source_csv,
        "source_rows": source_count,
        "min_aesthetic_score": args.min_aesthetic_score,
        "eligible_before_exclusion": eligible_before_exclusion,
        "excluded_sha256_count": len(excluded),
        "split": args.split,
        "seed": args.seed,
        "count": len(selected),
        "selection_digest": hashlib.sha256("\n".join(ordered_hashes).encode()).hexdigest(),
        "aesthetic_score": {
            "min": float(selected["aesthetic_score"].min()),
            "mean": float(selected["aesthetic_score"].mean()),
            "max": float(selected["aesthetic_score"].max()),
        },
        "sha256": ordered_hashes,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"metadata": str(output), **manifest}, indent=2))


if __name__ == "__main__":
    main()
