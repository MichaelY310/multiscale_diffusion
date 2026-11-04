"""Opt-in current-state correction of an already available coarse hidden grid.

This is a trainable candidate, not a training-free quality improvement. All new
attention/FFN work is performed on the native coarse grid. Current fine features
are pooled only to inform the correction; the fine reader still sees full fine
queries. No new positional encoding, cached attention, or cross-frame mask.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def pool_token_grid(x, fine_grid, coarse_grid):
    """Exact non-overlapping cell means on nested T/H/W grids (no interpolation)."""
    if len(fine_grid) != 3 or len(coarse_grid) != 3 or any(v <= 0 for v in (*fine_grid, *coarse_grid)):
        raise ValueError('Need positive T/H/W grids')
    if x.ndim != 3 or x.shape[1] != math.prod(fine_grid):
        raise ValueError('Fine sequence length/grid mismatch')
    if any(f % c for f, c in zip(fine_grid, coarse_grid)):
        raise ValueError('Require exactly nested token grids')
    b, _, h = x.shape
    ct, ch, cw = coarse_grid
    rt, rh, rw = (f // c for f, c in zip(fine_grid, coarse_grid))
    return x.reshape(b, ct, rt, ch, rh, cw, rw, h).mean((2, 4, 6)).reshape(b, -1, h)


class _ConditionedCoarseBlock(nn.Module):
    def __init__(self, hidden, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 4 * hidden))
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.out = nn.Linear(hidden, hidden)
        self.ffn = nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden))
        nn.init.zeros_(self.condition[-1].weight)
        nn.init.zeros_(self.condition[-1].bias)

    def forward(self, x, c):
        b, n, h = x.shape
        shift1, scale1, shift2, scale2 = self.condition(c).chunk(4, -1)
        z = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        q, k, v = self.qkv(z).reshape(b, n, 3, self.heads, h // self.heads).unbind(2)
        a = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2), dropout_p=0.)
        x = x + self.out(a.transpose(1, 2).reshape(b, n, h))
        z = self.norm2(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + self.ffn(z)


class CoarseMemoryRefiner(nn.Module):
    """Return memory + learned correction, exactly memory at initialization.

The final projection is zero-initialized but the complete branch always executes;
timings must include pooling, conditioning, coarse SDPA, FFNs and this projection.
The original memory is not overwritten or detached. Refinement is recomputed for
every fine timestep, never cached as though independent of the current state.
"""
    def __init__(self, hidden, heads, depth=2):
        super().__init__()
        if hidden % heads or depth not in (1, 2):
            raise ValueError('Unsupported hidden/heads or refinement depth')
        self.depth = depth
        self.memory_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.fine_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.fine_in = nn.Linear(hidden, hidden)
        self.blocks = nn.ModuleList([_ConditionedCoarseBlock(hidden, heads) for _ in range(depth)])
        self.output_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.output = nn.Linear(hidden, hidden)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, memory, fine, c, coarse_grid, fine_grid):
        b, _, h = fine.shape
        if memory.shape != (b, math.prod(coarse_grid), h) or c.shape != (b, h):
            raise ValueError('Native memory/current condition shape mismatch')
        with torch.autocast(fine.device.type, dtype=torch.bfloat16, enabled=fine.is_cuda):
            current = pool_token_grid(fine, fine_grid, coarse_grid)
            z = self.memory_norm(memory) + self.fine_in(self.fine_norm(current))
            for block in self.blocks:
                z = block(z, c)
            correction = self.output(self.output_norm(z))
        return (memory.float() + correction.float()).to(memory.dtype)
