"""Opt-in CURRENT-fine spatial-neighbor residual; not a generic speedup claim.

Use an explicit (T,H,W) token grid. Frames are independent: no convolution
along flattened token order or across frame boundaries. FP32 master weights,
BF16 AMP on CUDA, zero output projection for exact initial functional parity.
The whole branch executes even at initialization (no zero-weight shortcut).
"""
from __future__ import annotations

import math
import torch
from torch import nn


class FineSpatialNeighbor(nn.Module):
    def __init__(self, hidden, rank=128):
        super().__init__()
        if hidden <= 0 or rank <= 0:
            raise ValueError('hidden and rank must be positive')
        self.hidden, self.rank = hidden, rank
        self.norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.down = nn.Linear(hidden, rank, dtype=torch.float32)
        self.spatial = nn.Conv2d(rank, rank, 3, padding=1, groups=rank, dtype=torch.float32)
        self.activation = nn.SiLU()
        self.up = nn.Linear(rank, hidden, dtype=torch.float32)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, hidden, grid):
        if (len(grid) != 3 or any(type(n) is not int or n <= 0 for n in grid)
                or hidden.ndim != 3 or hidden.shape[1:] != (math.prod(grid), self.hidden)):
            raise ValueError('Expected [B,T*H*W,hidden] and explicit positive integer T/H/W grid')
        if hidden.dtype not in (torch.float32, torch.bfloat16):
            raise ValueError('FP32 or BF16 hidden required')
        if self.down.weight.dtype != torch.float32:
            raise ValueError('Keep correction master weights FP32; use AMP, not module.bfloat16()')
        batch = hidden.shape[0]
        frames, height, width = grid
        with torch.autocast(hidden.device.type, dtype=torch.bfloat16,
                            enabled=hidden.is_cuda and hidden.dtype == torch.bfloat16):
            small = self.down(self.norm(hidden.float()))
            small = small.reshape(batch * frames, height, width, self.rank).permute(0, 3, 1, 2).contiguous()
            small = self.activation(self.spatial(small))
            small = small.permute(0, 2, 3, 1).reshape(batch, math.prod(grid), self.rank)
            delta = self.up(small)
        return (hidden.float() + delta.float()).to(hidden.dtype)


def add_neighbor_after_bridge(bridge_output, neighbor, target_grid):
    """Separate helper to make the executed branch explicit in both domains."""
    return neighbor(bridge_output, target_grid)
