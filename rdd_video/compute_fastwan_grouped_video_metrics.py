#!/usr/bin/env python3
"""Same-prompt/seed SSIM and PSNR for the fair FastWan result groups.

These values measure pixel perturbation from each group's fixed-resolution
VSA reference.  They are deliberately not labeled as generative quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from compute_fastwan_fair_video_metrics import PSNR_RE, SSIM_RE, run_filter


def compare(reference: Path, candidate: Path) -> dict:
    ssim_matches = SSIM_RE.findall(run_filter(reference, candidate, "ssim"))
    psnr_matches = PSNR_RE.findall(run_filter(reference, candidate, "psnr"))
    if not ssim_matches or not psnr_matches:
        raise RuntimeError(f"could not parse ffmpeg metrics for {candidate}")
    return {
        "reference": reference.name,
        "candidate": candidate.name,
        "ssim_all": float(ssim_matches[-1]),
        "psnr_average_db": float(psnr_matches[-1]),
        "interpretation": "same prompt/seed pixel perturbation; not generative quality",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    root = args.result_dir.resolve()
    groups = {
        "primary_fox": (
            "final_fixed_vsa20_pre",
            ["final_temporal_vsa20", "final_temporal_c2f20", "final_temporal_c2f20_reuse2"],
        ),
        "quality_city": (
            "quality_city_fixed_vsa20",
            ["quality_city_temporal_vsa20", "quality_city_temporal_c2f20", "quality_city_temporal_c2f20_reuse2"],
        ),
        "quality_coast": (
            "quality_coast_fixed_vsa20",
            ["quality_coast_temporal_vsa20", "quality_coast_temporal_c2f20", "quality_coast_temporal_c2f20_reuse2"],
        ),
    }
    for tag in ("50", "25", "20", "12p5"):
        groups[f"density_{tag}"] = (
            f"pair_d{tag}_fixed_vsa_pre",
            [
                f"pair_d{tag}_temporal_vsa",
                f"pair_d{tag}_temporal_c2f_only",
                f"pair_d{tag}_temporal_c2f_reuse2",
            ],
        )

    output = {}
    for group, (reference_stem, candidate_stems) in groups.items():
        reference = root / f"{reference_stem}.mp4"
        output[group] = {}
        for stem in candidate_stems:
            result = compare(reference, root / f"{stem}.mp4")
            output[group][stem] = result
            print(group, stem, result)
    path = root / "grouped_same_seed_pixel_metrics.json"
    path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
