#!/usr/bin/env python3
"""Compare saved latent tensors from two fair FastWan experiment variants."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="before:after experiment stems; repeat for multiple comparisons",
    )
    args = parser.parse_args()
    pairs = [item.split(":", 1) for item in args.pair]
    for before, after in pairs:
        x = torch.load(args.root / f"{before}.latent.pt", map_location="cpu")
        y = torch.load(args.root / f"{after}.latent.pt", map_location="cpu")
        error = (x - y).abs()
        print(
            json.dumps(
                {
                    "before": before,
                    "after": after,
                    "max_abs": float(error.max()),
                    "mean_abs": float(error.mean()),
                    "exact": bool(torch.equal(x, y)),
                }
            )
        )


if __name__ == "__main__":
    main()
