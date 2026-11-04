"""Inference-only building block: nearest parent lift + residual addition.

No attention approximation. Input layout [B,N,H], map [Nfine]. The caller
computes tokenwise coarse transformations BEFORE lifting. Arbitrary bilinear
interpolation cannot commute with nonlinear activations this way.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _lift_add(C, F, P, O, NC: tl.constexpr, NF: tl.constexpr,
              H: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = i < TOTAL
    channel = i % H
    token = (i // H) % NF
    batch = i // (H * NF)
    parent = tl.load(P + token, mask=mask, other=0)
    low = tl.load(C + (batch * NC + parent) * H + channel, mask=mask, other=0).to(tl.float32)
    fine = tl.load(F + i, mask=mask, other=0).to(tl.float32)
    tl.store(O + i, low + fine, mask=mask)


def lift_add(coarse, fine, parent):
    # parent must be the in-range map produced by nearest_parent_map. Validate
    # external maps ONCE at setup, not with a device sync inside every call.
    if coarse.ndim != 3 or fine.ndim != 3 or coarse.shape[0] != fine.shape[0] or coarse.shape[2] != fine.shape[2]:
        raise ValueError("Expected compatible [B,N,H] tensors")
    if parent.shape != (fine.shape[1],) or parent.dtype not in (torch.int32, torch.int64):
        raise ValueError("Invalid parent map")
    if not coarse.is_cuda or coarse.device != fine.device or coarse.device != parent.device:
        raise ValueError("All inputs must be on the same CUDA device")
    if coarse.dtype != fine.dtype or fine.dtype not in (torch.float32,torch.bfloat16,torch.float16):
        raise ValueError("Require matching floating dtypes")
    if not all(x.is_contiguous() for x in (coarse, fine, parent)):
        raise ValueError("Require contiguous input layout")
    if torch.is_grad_enabled() and (coarse.requires_grad or fine.requires_grad):
        raise ValueError("Inference-only kernel; no autograd registration")
    out = torch.empty_like(fine)
    _lift_add[(triton.cdiv(fine.numel(), 256),)](coarse, fine, parent, out,
        coarse.shape[1], fine.shape[1], fine.shape[2], fine.numel(), 256)
    return out


def nearest_parent_map(coarse_grid, fine_grid, device):
    if len(coarse_grid) != 3 or len(fine_grid) != 3 or any(c < 1 or f < c for c, f in zip(coarse_grid, fine_grid)):
        raise ValueError("Expected positive coarse/fine 3D grids")
    t,h,w = torch.meshgrid(*(torch.arange(n,device=device) for n in fine_grid), indexing="ij")
    tc,hc,wc = (v*c//f for v,c,f in zip((t,h,w),coarse_grid,fine_grid))
    return ((tc*coarse_grid[1]+hc)*coarse_grid[2]+wc).reshape(-1).int().contiguous()
