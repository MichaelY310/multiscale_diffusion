"""Pack a directory of PNG images into arr_0 of an NPZ archive."""
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", help="Directory containing PNG images")
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--output", type=str, required=True, help="Output NPZ path")
    args = parser.parse_args()

    files = sorted([f for f in os.listdir(args.input_dir) if f.endswith(".png")])
    files = files[: args.num_samples]
    assert len(files) >= args.num_samples, f"Only found {len(files)} files, need {args.num_samples}"

    samples = []
    for f in tqdm(files, desc="Loading PNGs"):
        img = Image.open(os.path.join(args.input_dir, f)).convert("RGB")
        samples.append(np.asarray(img, dtype=np.uint8))

    arr = np.stack(samples)
    assert arr.shape == (args.num_samples, arr.shape[1], arr.shape[2], 3)
    np.savez(args.output, arr_0=arr)
    print(f"Saved {args.output} [{arr.shape}]")


if __name__ == "__main__":
    main()
