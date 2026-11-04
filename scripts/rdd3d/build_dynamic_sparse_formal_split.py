#!/usr/bin/env python3
"""Build a deterministic, leakage-resistant train/held-out SLat split.

Only assets with both a TRELLIS SLat and a usable conditioning render are
eligible.  The output directories contain relative symlinks, so the split is
cheap, auditable, and directly consumable by the dynamic-sparse RDD scripts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image


def usable_render(render_dir: Path, asset: str) -> tuple[bool, str | None]:
    root = render_dir / asset
    metadata_path = root / "transforms.json"
    if not metadata_path.is_file():
        return False, "missing transforms.json"
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        frames = metadata.get("frames", [])
        if not frames:
            return False, "empty frames"
        image_path = root / frames[0]["file_path"]
        with Image.open(image_path) as image:
            rgba = image.convert("RGBA")
            if rgba.getchannel("A").getbbox() is None:
                return False, "empty alpha mask"
    except Exception as exc:  # malformed upstream assets must not abort a split
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def usable_latent(path: Path, fine_resolution: int) -> tuple[bool, str | None]:
    try:
        with np.load(path) as data:
            if "coords" not in data or "feats" not in data:
                return False, "missing coords/feats"
            coords = data["coords"]
            feats = data["feats"]
        if coords.ndim != 2 or coords.shape[1] not in (3, 4):
            return False, f"invalid coords shape {coords.shape}"
        if coords.shape[1] == 4:
            coords = coords[:, 1:]
        if feats.ndim != 2 or len(feats) != len(coords):
            return False, f"invalid feats shape {feats.shape}"
        if not len(coords):
            return False, "empty latent"
        if np.any(coords < 0) or np.any(coords >= fine_resolution):
            return False, "coordinates outside fine-resolution grid"
        if len(np.unique(coords.astype(np.int64), axis=0)) != len(coords):
            return False, "duplicate sparse coordinates"
        if not np.isfinite(feats).all():
            return False, "non-finite latent features"
    except Exception as exc:
        return False, f"latent {type(exc).__name__}: {exc}"
    return True, None


def symlink_assets(paths: list[Path], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    expected = {path.name for path in paths}
    unexpected = {
        path.name for path in destination.glob("*.npz") if path.name not in expected
    }
    if unexpected:
        names = ", ".join(sorted(unexpected)[:5])
        raise RuntimeError(
            f"{destination} contains files outside this deterministic split: {names}"
        )
    for source in paths:
        link = destination / source.name
        if link.exists() or link.is_symlink():
            if link.resolve() != source.resolve():
                raise RuntimeError(f"existing link points elsewhere: {link}")
            continue
        link.symlink_to(source.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--heldout-count", type=int, default=32)
    parser.add_argument(
        "--dev-count",
        type=int,
        default=0,
        help="optional development split; kept separate from the final test split",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=0,
        help="optional untouched final-test split",
    )
    parser.add_argument("--seed", type=int, default=98317)
    parser.add_argument("--fine-resolution", type=int, default=32)
    parser.add_argument(
        "--exclude-latent-dir",
        type=Path,
        action="append",
        default=[],
        help="exclude assets used by earlier pilot runs; may be repeated",
    )
    args = parser.parse_args()

    excluded = set()
    for directory in args.exclude_latent_dir:
        excluded.update(path.stem for path in directory.glob("*.npz"))

    rejected: dict[str, str] = {}
    eligible: list[Path] = []
    for path in sorted(args.latent_dir.glob("*.npz")):
        if path.stem in excluded:
            rejected[path.stem] = "excluded prior pilot asset"
            continue
        usable, reason = usable_latent(path, args.fine_resolution)
        if not usable:
            rejected[path.stem] = reason or "unusable latent"
            continue
        usable, reason = usable_render(args.render_dir, path.stem)
        if usable:
            eligible.append(path)
        else:
            rejected[path.stem] = reason or "unusable render"

    for name in ("train_count", "heldout_count", "dev_count", "test_count"):
        if getattr(args, name) < 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    required = (
        args.train_count
        + args.heldout_count
        + args.dev_count
        + args.test_count
    )
    if len(eligible) < required:
        raise SystemExit(
            f"need {required} eligible assets, found {len(eligible)} "
            f"({len(rejected)} rejected)"
        )
    random.Random(args.seed).shuffle(eligible)
    cursor = 0

    def take(count: int) -> list[Path]:
        nonlocal cursor
        selected = sorted(eligible[cursor : cursor + count], key=lambda path: path.name)
        cursor += count
        return selected

    train = take(args.train_count)
    heldout = take(args.heldout_count)
    dev = take(args.dev_count)
    final_test = take(args.test_count)
    named_splits = {
        "train": train,
        "heldout": heldout,
        "dev": dev,
        "final_test": final_test,
    }
    seen: set[str] = set()
    for name, paths in named_splits.items():
        assets = {path.stem for path in paths}
        overlap = seen & assets
        if overlap:
            raise AssertionError(f"{name} overlap: {sorted(overlap)[:3]}")
        seen.update(assets)

    symlink_assets(train, args.output_root / "train_latents")
    symlink_assets(heldout, args.output_root / "heldout_latents")
    if args.dev_count:
        symlink_assets(dev, args.output_root / "dev_latents")
    if args.test_count:
        symlink_assets(final_test, args.output_root / "final_test_latents")
    symlink_assets(
        train + heldout + dev + final_test,
        args.output_root / "all_latents",
    )
    manifest = {
        "latent_dir": str(args.latent_dir.resolve()),
        "render_dir": str(args.render_dir.resolve()),
        "seed": args.seed,
        "train_count": len(train),
        "heldout_count": len(heldout),
        "dev_count": len(dev),
        "test_count": len(final_test),
        "eligible_count": len(eligible),
        "excluded_asset_count": len(excluded),
        "train_assets": [path.stem for path in train],
        "heldout_assets": [path.stem for path in heldout],
        "dev_assets": [path.stem for path in dev],
        "final_test_assets": [path.stem for path in final_test],
        "unused_eligible_assets": [path.stem for path in eligible[required:]],
        "rejected_assets": rejected,
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in (
        "train_count", "heldout_count", "dev_count", "test_count",
        "eligible_count", "excluded_asset_count"
    )}, indent=2))


if __name__ == "__main__":
    main()
