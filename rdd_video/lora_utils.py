"""Small dependency-free LoRA helpers for the official Wan transformer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import nn


class LoRALinear(nn.Module):
    """Frozen ``nn.Linear`` plus a trainable low-rank residual."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        factory_kwargs = {"device": base.weight.device, "dtype": base.weight.dtype}
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False, **factory_kwargs)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False, **factory_kwargs)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_b(self.lora_a(self.dropout(x))) * self.scaling

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        merged = nn.Linear(
            self.base.in_features,
            self.base.out_features,
            bias=self.base.bias is not None,
            device=self.base.weight.device,
            dtype=self.base.weight.dtype,
        )
        delta = self.lora_b.weight @ self.lora_a.weight
        merged.weight.copy_(self.base.weight + delta.to(self.base.weight.dtype) * self.scaling)
        if self.base.bias is not None:
            merged.bias.copy_(self.base.bias)
        return merged


DEFAULT_WAN_LORA_PATTERNS = (
    r"^blocks\.\d+\.self_attn\.(q|k|v|o)$",
    r"^blocks\.\d+\.cross_attn\.(q|k|v|o|k_img|v_img)$",
    r"^blocks\.\d+\.ffn\.(0|2)$",
)


@dataclass(frozen=True)
class LoRAInjectionReport:
    names: tuple[str, ...]
    trainable_parameters: int


def _matches(name: str, patterns: Sequence[str]) -> bool:
    return any(re.search(pattern, name) is not None for pattern in patterns)


def inject_lora(
    model: nn.Module,
    *,
    rank: int = 128,
    alpha: float = 128.0,
    dropout: float = 0.0,
    patterns: Sequence[str] = DEFAULT_WAN_LORA_PATTERNS,
) -> LoRAInjectionReport:
    """Freeze a model and replace selected Wan linears with LoRA wrappers."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    replacements: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _matches(name, patterns):
            replacements.append((name, module))
    if not replacements:
        sample = [name for name, module in model.named_modules() if isinstance(module, nn.Linear)][:20]
        raise RuntimeError(
            "No Wan linear layer matched the LoRA target patterns. First linear names: "
            + ", ".join(sample)
        )

    names: list[str] = []
    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        replacement = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, child_name, replacement)
        names.append(name)

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return LoRAInjectionReport(tuple(names), trainable)


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return only portable LoRA matrices (CPU tensors)."""

    return {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if ".lora_a." in name or ".lora_b." in name
    }


def load_lora_state_dict(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    result = model.load_state_dict(state, strict=False)
    unexpected = [key for key in result.unexpected_keys if "lora_" in key]
    missing = [key for key in result.missing_keys if "lora_" in key]
    if unexpected or missing:
        raise RuntimeError(f"LoRA checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def merge_lora_(model: nn.Module) -> int:
    """Merge every LoRA residual into its base linear in-place."""

    replacements: list[tuple[str, LoRALinear]] = [
        (name, module) for name, module in model.named_modules() if isinstance(module, LoRALinear)
    ]
    for name, module in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, module.merged_linear())
    return len(replacements)
