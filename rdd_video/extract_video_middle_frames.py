"""Extract a representative middle frame from each MP4 in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v3 as iio
import imageio.v2 as iio_v2
from PIL import Image


def main(directory: Path) -> None:
    for video_path in sorted(directory.glob("*.mp4")):
        try:
            frames = list(iio.imiter(video_path, plugin="pyav"))
            frame = frames[len(frames) // 2]
            frame_count = len(frames)
        except ImportError:
            # imageio-ffmpeg bundles its own executable and is present in the
            # Wan environment even when the system has no ffmpeg command.
            reader = iio_v2.get_reader(video_path, format="ffmpeg")
            try:
                frame_count = reader.count_frames()
                frame = reader.get_data(frame_count // 2)
            finally:
                reader.close()
        if frame_count <= 0:
            raise RuntimeError(f"no frames decoded from {video_path}")
        output = video_path.with_name(f"{video_path.stem}_mid.jpg")
        Image.fromarray(frame).save(output, quality=88)
        print(f"{video_path.name}: {frame_count} frames -> {output.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    main(args.directory)
