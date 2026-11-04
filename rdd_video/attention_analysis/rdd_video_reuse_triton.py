"""Small fused kernel for deployable RDD attention-output reuse.

Full attention-probability reuse would cache O(H*N^2*density) values per
layer.  At the Wan2.1 video grid that is too large to keep for all 30 layers.
The deployable alternative, as in attention broadcast/cache methods, stores the
post-attention output O(N*D) and on cache hits skips QKV/QK/softmax/AV/output
projection.  This kernel fuses the cached branch's gate and residual update.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _cached_output_residual_kernel(x, cached, out, n_elements, gate: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n_elements
    x_value = tl.load(x + offsets, mask=mask).to(tl.float32)
    cached_value = tl.load(cached + offsets, mask=mask).to(tl.float32)
    tl.store(out + offsets, x_value + gate * cached_value, mask=mask)


def cached_output_residual(x: torch.Tensor, cached: torch.Tensor, gate: float = 1.0) -> torch.Tensor:
    if x.shape != cached.shape or x.dtype != cached.dtype or x.device != cached.device:
        raise ValueError("x and cached must have identical shape/dtype/device")
    if not x.is_contiguous() or not cached.is_contiguous():
        raise ValueError("x and cached must be contiguous")
    out = torch.empty_like(x)
    block = 256
    _cached_output_residual_kernel[(triton.cdiv(x.numel(), block),)](
        x, cached, out, x.numel(), gate=gate, block=block
    )
    return out

