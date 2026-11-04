"""Compute same-seed fidelity-to-baseline proxies for the FastWan sweep.

SSIM/PSNR do not measure prompt quality.  Here they answer the narrower
engineering question: how much did an acceleration setting perturb the pixels
of the fixed-resolution VSA20 reference under the same checkpoint/prompt/seed?
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import shutil
from pathlib import Path


NUMBER = r"(?:[0-9.eE+-]+|inf)"
SSIM_RE = re.compile(rf"All:({NUMBER})")
PSNR_RE = re.compile(rf"average:({NUMBER})")


def run_filter(reference: Path, candidate: Path, filter_name: str) -> str:
    executable = shutil.which("ffmpeg")
    if executable is None:
        import imageio_ffmpeg

        executable = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        executable,
        "-hide_banner",
        "-i",
        str(reference),
        "-i",
        str(candidate),
        "-lavfi",
        filter_name,
        "-f",
        "null",
        "-",
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode:
        raise RuntimeError(f"ffmpeg {filter_name} failed for {candidate}:\n{result.stdout}")
    return result.stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--reference", default="fixed_vsa20.mp4")
    args = parser.parse_args()
    root = args.result_dir.resolve()
    reference = root / args.reference
    if not reference.is_file():
        raise FileNotFoundError(reference)

    rows = {}
    for candidate in sorted(root.glob("*.mp4")):
        ssim_text = run_filter(reference, candidate, "ssim")
        psnr_text = run_filter(reference, candidate, "psnr")
        ssim_matches = SSIM_RE.findall(ssim_text)
        psnr_matches = PSNR_RE.findall(psnr_text)
        if not ssim_matches or not psnr_matches:
            raise RuntimeError(f"could not parse ffmpeg metrics for {candidate}")
        rows[candidate.stem] = {
            "reference": reference.name,
            "ssim_all": float(ssim_matches[-1]),
            "psnr_average_db": float(psnr_matches[-1]),
            "interpretation": "same-seed pixel fidelity proxy; not a generative-quality metric",
        }
        print(candidate.stem, rows[candidate.stem])
    output = root / "same_seed_pixel_metrics.json"
    output.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
