"""Build a deterministic, action-rich prompt bank for Wan pilot fine-tuning.

This is a reproducible synthetic-data pilot, not a replacement for a licensed
large-scale video-caption corpus. Final evaluation must use a disjoint standard
suite such as VBench prompts.
"""

import argparse
import itertools
import json
import random
from pathlib import Path


SUBJECTS = [
    "a red fox", "a golden retriever", "a hummingbird", "a sea turtle",
    "an astronaut", "a ceramic artist", "a street dancer", "a mountain climber",
    "a vintage motorcycle", "a red sailboat", "a steam locomotive", "a small robot",
    "a paper dragon", "a glass sculpture", "a bowl of fruit", "a field of sunflowers",
]
ACTIONS = [
    "moving steadily through the scene", "turning toward the camera",
    "crossing from left to right", "accelerating and then slowing down",
    "interacting with nearby objects", "emerging from behind an obstacle",
    "spinning in place", "jumping over a small barrier",
]
SCENES = [
    "in a misty forest", "on a windswept beach", "in a neon-lit city at night",
    "inside a sunlit workshop", "across a snowy mountain ridge",
    "beside a quiet alpine lake", "in a crowded outdoor market",
    "underwater above a coral reef", "in a desert during golden hour",
    "inside a futuristic space station", "on a rainy cobblestone street",
    "in a meadow filled with wildflowers",
]
CAMERAS = [
    "slow dolly-in", "smooth tracking shot", "wide static shot",
    "gentle handheld camera", "low-angle follow shot", "slow orbiting camera",
]
LIGHTING = [
    "soft natural light", "dramatic rim lighting", "overcast cinematic light",
    "warm sunset light", "cool moonlight", "high-contrast studio lighting",
]


def build_prompts(count: int, seed: int):
    combinations = list(itertools.product(SUBJECTS, ACTIONS, SCENES, CAMERAS, LIGHTING))
    rng = random.Random(seed)
    rng.shuffle(combinations)
    if count > len(combinations):
        raise ValueError(f"Requested {count} prompts but only {len(combinations)} are unique")
    return [
        f"{subject} {action} {scene}, {camera}, {lighting}, coherent motion, realistic details"
        for subject, action, scene, camera, lighting in combinations[:count]
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("rdd_video_data/prompts"))
    parser.add_argument("--train-count", type=int, default=2000)
    parser.add_argument("--eval-count", type=int, default=256)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    prompts = build_prompts(args.train_count + args.eval_count, args.seed)
    train = prompts[: args.train_count]
    evaluation = prompts[args.train_count :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "train_prompts.txt").write_text("\n".join(train) + "\n", encoding="utf-8")
    (args.output_dir / "pilot_eval_prompts.txt").write_text(
        "\n".join(evaluation) + "\n", encoding="utf-8"
    )
    metadata = {
        "seed": args.seed,
        "train_count": len(train),
        "eval_count": len(evaluation),
        "note": "Pilot synthetic prompts; use official VBench prompts for final reporting.",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote {len(train)} train and {len(evaluation)} disjoint pilot-eval prompts")


if __name__ == "__main__":
    main()

