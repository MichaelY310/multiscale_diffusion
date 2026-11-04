#!/usr/bin/env python3
"""Create compact previews of attention-tuning comparison grids."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("roots", nargs="+", type=Path)
    parser.add_argument("--max-width", type=int, default=2048)
    parser.add_argument("--quality", type=int, default=90)
    args = parser.parse_args()

    names = {
        "centroid_compensation_comparison.png",
        "adaptive_probability_cache_comparison.png",
    }
    for root in args.roots:
        for source in sorted(root.glob("seed_*/*.png")):
            if source.name not in names:
                continue
            with Image.open(source) as image:
                image = image.convert("RGB")
                if image.width > args.max_width:
                    height = round(image.height * args.max_width / image.width)
                    image = image.resize((args.max_width, height), Image.Resampling.LANCZOS)
                target = source.with_name(source.stem + "_preview.webp")
                image.save(target, "WEBP", quality=args.quality, method=6)
                print(f"saved {target}")


if __name__ == "__main__":
    main()
