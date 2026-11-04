"""Create compact six-frame contact sheets for every MP4 in a directory."""

from __future__ import annotations

import argparse
from pathlib import Path

import imageio.v2 as iio
from PIL import Image, ImageDraw


def make_sheet(video_path: Path, output: Path) -> None:
    reader = iio.get_reader(video_path, format="ffmpeg")
    try:
        frame_count = reader.count_frames()
        indices = [round(i * (frame_count - 1) / 5) for i in range(6)]
        frames = [Image.fromarray(reader.get_data(index)) for index in indices]
    finally:
        reader.close()

    thumb_width = 416
    thumb_height = round(frames[0].height * thumb_width / frames[0].width)
    label_height = 24
    sheet = Image.new("RGB", (thumb_width * 3, (thumb_height + label_height) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    for slot, (index, frame) in enumerate(zip(indices, frames)):
        row, column = divmod(slot, 3)
        x = column * thumb_width
        y = row * (thumb_height + label_height)
        frame = frame.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        sheet.paste(frame, (x, y + label_height))
        draw.text((x + 6, y + 4), f"frame {index}", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=82, optimize=True)
    print(f"{video_path.name} -> {output.name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--stems",
        default="",
        help="Optional comma-separated video stems; default processes every MP4.",
    )
    args = parser.parse_args()
    if args.stems:
        video_paths = [
            args.directory / f"{stem.strip()}.mp4"
            for stem in args.stems.split(",")
            if stem.strip()
        ]
    else:
        video_paths = sorted(args.directory.glob("*.mp4"))
    for video_path in video_paths:
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        make_sheet(video_path, video_path.with_name(f"{video_path.stem}_contact.jpg"))


if __name__ == "__main__":
    main()
