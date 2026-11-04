#!/usr/bin/env python3
"""Cache the public DINOv2-L/14 weights used by the ungated smoke test."""

import torch


def main() -> None:
    model = torch.hub.load(
        "facebookresearch/dinov2",
        "dinov2_vitl14",
        pretrained=True,
        trust_repo=True,
    )
    print(type(model).__name__)


if __name__ == "__main__":
    main()
