"""
Download a single file from a HuggingFace model repository.

Usage:
  python scripts/hf_download.py \
      --repo  your-username/your-repo-name \
      --file  checkpoints_32x32/0068000.pt \
      --out   /path/to/save/0068000.pt       # optional, defaults to current dir
"""

import argparse
import os
from huggingface_hub import hf_hub_download


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="HuggingFace repo id, e.g. username/my-model")
    parser.add_argument("--file", required=True, help="Path of the file inside the repo")
    parser.add_argument("--out", default=None, help="Where to save the file (default: current directory)")
    args = parser.parse_args()

    out_dir = args.out or os.getcwd()
    os.makedirs(out_dir, exist_ok=True)

    print(f"Downloading {args.repo}/{args.file} → {out_dir}")
    local_path = hf_hub_download(
        repo_id=args.repo,
        filename=args.file,
        local_dir=out_dir,
    )
    print(f"Saved to: {local_path}")


if __name__ == "__main__":
    main()
