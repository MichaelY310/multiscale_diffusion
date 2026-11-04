#!/usr/bin/env python3
"""Download a matched early/late Unified RDD checkpoint pair for smoke tests.

These defaults are intended to verify backward-compatible sampling features
(legacy CFG, limited-interval guidance, and Autoguidance).  They are not a
claim that the checkpoints use the final paper's ImageNet data protocol.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id", default="yang335/search_agent_hl_to_i18n_transport"
    )
    parser.add_argument(
        "--output-dir",
        default=".tmp/rdd_sampling_ckpts",
        help="hf_hub_download local_dir; repository subdirectories are retained.",
    )
    parser.add_argument("--model-family", default="RDD-Unified-L")
    parser.add_argument("--good-step", default="0400000")
    parser.add_argument("--weak-step", default="0100000")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, step in (("good", args.good_step), ("weak", args.weak_step)):
        filename = f"checkpoints/{args.model_family}/{step}.pt"
        path = hf_hub_download(
            repo_id=args.repo_id,
            filename=filename,
            repo_type="model",
            local_dir=output_dir,
        )
        print(f"{label.upper()}_CKPT={path}", flush=True)


if __name__ == "__main__":
    main()
