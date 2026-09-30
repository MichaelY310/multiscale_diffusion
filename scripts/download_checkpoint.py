"""Download one archived RDD checkpoint from Hugging Face."""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        default="yang335/search_agent_hl_to_i18n_transport",
        help="Hugging Face model repository",
    )
    parser.add_argument(
        "--filename",
        default="checkpoints/RDD-Unified-L/0400000.pt",
        help="Repository-relative checkpoint path",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Local directory; repository subdirectories are preserved",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=args.repo,
        repo_type="model",
        filename=args.filename,
        local_dir=output_dir,
    )
    print(path)


if __name__ == "__main__":
    main()
