"""Build the paper-scale Wan teacher prompt bank from public Open-Sora-Plan captions.

The PyramidalWan paper uses 80K Wan2.1-14B synthetic videos and draws prompts
from the Open-Sora-Plan data source.  The exact private 350K subset is not
released.  This script therefore makes the closest reproducible replacement:
80K unique captions from the public MIT-licensed v1.1 high-quality annotation
files, with a fixed seed and a provenance manifest.
"""

import argparse
import hashlib
import json
import random
from pathlib import Path

from huggingface_hub import hf_hub_download


DEFAULT_ANNOTATIONS = (
    "anno_json/v1.1.0_HQ_part1.json",
    "anno_json/v1.1.0_HQ_part2.json",
    "anno_json/v1.1.0_HQ_part3.json",
)


def normalize_caption(value):
    if isinstance(value, list):
        value = value[0] if value else ""
    if not isinstance(value, str):
        return ""
    return " ".join(value.strip().split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default="LanguageBind/Open-Sora-Plan-v1.2.0",
    )
    parser.add_argument("--revision", default="main")
    parser.add_argument("--count", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("rdd_video_data/prompts/opensora_hq_train_80k.txt"),
    )
    parser.add_argument(
        "--exclude-file",
        type=Path,
        default=None,
        help="Optional newline-separated evaluation prompts to exclude exactly.",
    )
    parser.add_argument(
        "--annotation",
        action="append",
        dest="annotations",
        help="Override annotation filename; may be supplied more than once.",
    )
    args = parser.parse_args()

    excluded = set()
    if args.exclude_file:
        excluded = {
            " ".join(line.strip().split())
            for line in args.exclude_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }

    annotations = tuple(args.annotations or DEFAULT_ANNOTATIONS)
    captions = set()
    resolved_files = []
    for filename in annotations:
        local_path = hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=filename,
            revision=args.revision,
        )
        resolved_files.append(str(local_path))
        with open(local_path, "r", encoding="utf-8") as handle:
            records = json.load(handle)
        if not isinstance(records, list):
            raise TypeError(f"Expected a JSON list in {filename}")
        for record in records:
            if not isinstance(record, dict):
                continue
            caption = normalize_caption(record.get("cap", record.get("caption", "")))
            if 20 <= len(caption) <= 600 and caption not in excluded:
                captions.add(caption)

    if len(captions) < args.count:
        raise RuntimeError(
            f"Only {len(captions)} unique captions remain, fewer than --count={args.count}."
        )

    population = sorted(captions)
    rng = random.Random(args.seed)
    rng.shuffle(population)
    selected = population[: args.count]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(selected) + "\n"
    args.output.write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    manifest = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "annotation_files": list(annotations),
        "resolved_cache_files": resolved_files,
        "seed": args.seed,
        "available_unique_captions": len(captions),
        "selected_count": len(selected),
        "excluded_exact_prompts": len(excluded),
        "sha256": digest,
        "note": (
            "Reproducible public proxy for PyramidalWan's unreleased 350K "
            "Open-Sora-Plan subset; use these prompts to generate 80K Wan2.1-14B videos."
        ),
    }
    args.output.with_suffix(args.output.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(selected)} prompts to {args.output} (sha256={digest})")


if __name__ == "__main__":
    main()

