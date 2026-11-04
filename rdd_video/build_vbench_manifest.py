"""Create a deterministic generation manifest for the official VBench suite."""

import argparse
import json
from collections import OrderedDict
from pathlib import Path


DIMENSIONS = [
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "object_class",
    "multiple_objects",
    "human_action",
    "color",
    "spatial_relationship",
    "scene",
    "temporal_style",
    "appearance_style",
    "overall_consistency",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vbench-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--samples-per-prompt", type=int, default=5)
    parser.add_argument("--temporal-flickering-samples", type=int, default=25)
    parser.add_argument(
        "--dimensions",
        default=",".join(DIMENSIONS),
        help="Comma-separated VBench dimensions.",
    )
    args = parser.parse_args()

    dimensions = [item.strip() for item in args.dimensions.split(",") if item.strip()]
    unknown = sorted(set(dimensions) - set(DIMENSIONS))
    if unknown:
        raise ValueError(f"Unknown VBench dimensions: {unknown}")

    prompts = OrderedDict()
    prompt_root = args.vbench_dir / "prompts" / "prompts_per_dimension"
    for dimension in dimensions:
        path = prompt_root / f"{dimension}.txt"
        if not path.is_file():
            raise FileNotFoundError(path)
        repeat_count = (
            args.temporal_flickering_samples
            if dimension == "temporal_flickering"
            else args.samples_per_prompt
        )
        for line in path.read_text(encoding="utf-8").splitlines():
            prompt = line.strip()
            if not prompt:
                continue
            if "/" in prompt or "\\" in prompt or "\x00" in prompt:
                raise ValueError(
                    f"Prompt cannot be represented by VBench's literal filename convention: {prompt!r}"
                )
            entry = prompts.setdefault(prompt, {"repeats": 0, "dimensions": []})
            entry["repeats"] = max(entry["repeats"], repeat_count)
            if dimension not in entry["dimensions"]:
                entry["dimensions"].append(dimension)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    record_index = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for prompt, info in prompts.items():
            for sample_index in range(info["repeats"]):
                record = {
                    "prompt": prompt,
                    "sample_index": sample_index,
                    "seed": args.base_seed + record_index,
                    "relative_path": f"{prompt}-{sample_index}.mp4",
                    "dimensions": info["dimensions"],
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                record_index += 1

    manifest = {
        "vbench_dir": str(args.vbench_dir.resolve()),
        "dimensions": dimensions,
        "unique_prompts": len(prompts),
        "videos": record_index,
        "base_seed": args.base_seed,
        "samples_per_prompt": args.samples_per_prompt,
        "temporal_flickering_samples": args.temporal_flickering_samples,
    }
    args.output.with_suffix(args.output.suffix + ".meta.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"Wrote {record_index} videos for {len(prompts)} unique official VBench prompts "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()

