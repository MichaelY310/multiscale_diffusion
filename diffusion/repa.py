"""Minimal REPA teacher utilities for RDD ImageNet training.

This module follows the public REPA recipe with a frozen DINOv2 patch-token
teacher.  RDD extends the alignment across its resolution stages by pooling the
same clean-image teacher grid to each stage's token grid.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch
import torch.nn.functional as F


DINOV2_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


def load_state_dict_with_optional_repa(
    module: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    allow_missing_repa: bool,
) -> None:
    """Load a legacy or REPA checkpoint without hiding unrelated mismatches.

    A legacy RDD checkpoint naturally lacks the newly introduced
    ``repa_projector`` parameters.  ``strict=False`` alone would also ignore a
    misspelled layer, wrong architecture, or unexpected checkpoint contents.
    This helper permits only the intentional projector gap and keeps every
    other state-dict mismatch fatal.
    """

    if not allow_missing_repa:
        module.load_state_dict(state_dict, strict=True)
        return

    incompatible = module.load_state_dict(state_dict, strict=False)
    invalid_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith("repa_projector.")
    ]
    if invalid_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "Checkpoint mismatch outside the optional REPA projector: "
            f"missing={invalid_missing}, unexpected={incompatible.unexpected_keys}"
        )


def dinov2_dimension(model_name: str) -> int:
    try:
        return DINOV2_DIMS[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported DINOv2 teacher {model_name!r}; choose one of {sorted(DINOV2_DIMS)}"
        ) from exc


def load_dinov2_teacher(
    model_name: str,
    device: torch.device,
    *,
    repo_or_dir: Optional[str] = None,
    weights_path: Optional[str] = None,
) -> torch.nn.Module:
    """Load and freeze a DINOv2 teacher.

    ``repo_or_dir`` enables fully offline runs after cloning the official
    facebookresearch/dinov2 repository.  If omitted, torch.hub uses its cache
    and downloads the official source when necessary.  ``weights_path`` is an
    optional raw state_dict/checkpoint override for air-gapped servers.
    """

    if repo_or_dir:
        repo = Path(repo_or_dir)
        if not repo.exists():
            raise FileNotFoundError(f"DINOv2 repository not found: {repo}")
        teacher = torch.hub.load(str(repo), model_name, source="local", pretrained=not weights_path)
    else:
        teacher = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            source="github",
            pretrained=not weights_path,
        )

    if weights_path:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        state_dict = {
            key.removeprefix("module."): value for key, value in state_dict.items()
        }
        teacher.load_state_dict(state_dict, strict=True)

    teacher = teacher.to(device).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    return teacher


@torch.no_grad()
def dinov2_patch_tokens(teacher: torch.nn.Module, images_minus1_to1: torch.Tensor) -> torch.Tensor:
    """Extract normalized DINOv2 patch tokens from the exact cropped image batch."""

    images = (images_minus1_to1.float() + 1.0) * 0.5
    images = F.interpolate(images, size=(224, 224), mode="bicubic", align_corners=False, antialias=True)
    mean = torch.tensor((0.485, 0.456, 0.406), device=images.device).view(1, 3, 1, 1)
    std = torch.tensor((0.229, 0.224, 0.225), device=images.device).view(1, 3, 1, 1)
    outputs = teacher.forward_features((images - mean) / std)
    if isinstance(outputs, dict):
        if "x_norm_patchtokens" not in outputs:
            raise KeyError(
                "DINOv2 forward_features output lacks 'x_norm_patchtokens': "
                f"{sorted(outputs)}"
            )
        tokens = outputs["x_norm_patchtokens"]
    else:
        tokens = outputs
        # Some compatible ViTs include a class token in a raw tensor output.
        side = int(round((tokens.shape[1] - 1) ** 0.5))
        if side * side == tokens.shape[1] - 1:
            tokens = tokens[:, 1:]
    return tokens.detach()
