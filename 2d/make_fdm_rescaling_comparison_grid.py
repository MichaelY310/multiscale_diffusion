#!/usr/bin/env python3
"""Create a same-seed three-column grid for the f-DM rescaling probe."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--sp", type=Path, required=True)
    parser.add_argument("--vp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-rows", type=int, default=8)
    args = parser.parse_args()

    folders = [args.original, args.sp, args.vp]
    labels = [
        "Original RDD (no rescale)",
        "f-DM SP (inference probe)",
        "f-DM VP (inference probe)",
    ]
    files_by_folder = [
        {path.name: path for path in folder.glob("*.png")} for folder in folders
    ]
    names = sorted(set.intersection(*(set(files) for files in files_by_folder)))
    names = names[: args.max_rows]
    if not names:
        raise RuntimeError("No matching PNG filenames found in all three folders")

    with Image.open(files_by_folder[0][names[0]]) as example:
        tile_w, tile_h = example.size
    label_h = 30
    sample_h = 20
    canvas = Image.new(
        "RGB",
        (3 * tile_w, label_h + len(names) * (tile_h + sample_h)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column, label in enumerate(labels):
        draw.text((column * tile_w + 6, 8), label, fill="black", font=font)

    for row, name in enumerate(names):
        y = label_h + row * (tile_h + sample_h)
        draw.text((6, y + 3), f"same seed / sample {name}", fill="#555555", font=font)
        for column, files in enumerate(files_by_folder):
            with Image.open(files[name]) as image:
                canvas.paste(image.convert("RGB"), (column * tile_w, y + sample_h))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output)
    print(f"Saved {len(names)} same-seed rows to {args.output}")


if __name__ == "__main__":
    main()
