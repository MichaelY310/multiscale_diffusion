#!/usr/bin/env python3
"""Build a paired contact sheet for two stage-transition sample folders."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--left-label", default="Noise filling (preserve coarse noise)")
    parser.add_argument("--right-label", default="Redraw (fresh full noise)")
    parser.add_argument("--pairs-per-row", type=int, default=4)
    args = parser.parse_args()

    left_files = {path.name: path for path in args.left.glob("*.png")}
    right_files = {path.name: path for path in args.right.glob("*.png")}
    names = sorted(left_files.keys() & right_files.keys())
    if not names:
        raise RuntimeError("No matching PNG filenames found")
    if args.pairs_per_row <= 0:
        raise ValueError("--pairs-per-row must be positive")

    with Image.open(left_files[names[0]]) as example:
        tile_w, tile_h = example.size
    header_h = 34
    pair_w = tile_w * 2
    rows = math.ceil(len(names) / args.pairs_per_row)
    canvas = Image.new(
        "RGB",
        (pair_w * args.pairs_per_row, rows * (tile_h + header_h)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, name in enumerate(names):
        row, col = divmod(index, args.pairs_per_row)
        x = col * pair_w
        y = row * (tile_h + header_h)
        draw.text((x + 5, y + 3), args.left_label, fill="black", font=font)
        draw.text((x + tile_w + 5, y + 3), args.right_label, fill="black", font=font)
        draw.text((x + 5, y + 18), f"sample {name}", fill="#555555", font=font)
        with Image.open(left_files[name]) as left:
            canvas.paste(left.convert("RGB"), (x, y + header_h))
        with Image.open(right_files[name]) as right:
            canvas.paste(right.convert("RGB"), (x + tile_w, y + header_h))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved {len(names)} paired samples to {args.output}")


if __name__ == "__main__":
    main()
