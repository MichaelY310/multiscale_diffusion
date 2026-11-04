#!/usr/bin/env python3
"""Build labeled 4x4 grids from two deterministic preview directories."""

from argparse import ArgumentParser
from pathlib import Path

from PIL import Image, ImageDraw


def make_grid(directory: Path, title: str, count: int = 16) -> Image.Image:
    paths = sorted(directory.glob("[0-9]*.png"))[:count]
    if len(paths) != count:
        raise RuntimeError(f"expected {count} images in {directory}, found {len(paths)}")
    grid = Image.new("RGB", (1024, 1064), "black")
    ImageDraw.Draw(grid).text((10, 12), title, fill="white")
    for index, path in enumerate(paths):
        image = Image.open(path).convert("RGB").resize((256, 256))
        grid.paste(image, ((index % 4) * 256, 40 + (index // 4) * 256))
    return grid


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--left-dir", type=Path, required=True)
    parser.add_argument("--right-dir", type=Path, required=True)
    parser.add_argument("--left-title", required=True)
    parser.add_argument("--right-title", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    left = make_grid(args.left_dir, args.left_title)
    right = make_grid(args.right_dir, args.right_title)
    left.save(args.left_dir / "grid_4x4.png")
    right.save(args.right_dir / "grid_4x4.png")
    comparison = Image.new("RGB", (2048, 1064), "black")
    comparison.paste(left, (0, 0))
    comparison.paste(right, (1024, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(args.output)


if __name__ == "__main__":
    main()
