#!/usr/bin/env python3
"""Regenerate polished frequency figures from completed JSON/CSV statistics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import analyze_image_frequency as frequency


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.analysis_dir.expanduser().resolve()
    payload = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    with (root / "fft_radial_profiles.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        parser.error("empty fft_radial_profiles.csv")
    centers = np.asarray([float(row["frequency"]) for row in rows], dtype=np.float64)
    labels = [item["label"] for item in payload["datasets"]]
    profiles = {
        label: np.asarray([float(row[label]) for row in rows], dtype=np.float64)
        for label in labels
    }
    frequency._write_plots(payload["datasets"], profiles, centers, root)
    print(f"[done] polished plots in {root}", flush=True)


if __name__ == "__main__":
    main()
