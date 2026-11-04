"""Fuse nearest coarse FFN transport and current fine neuron contribution."""
import torch
import triton
import triton.language as tl


@triton.jit
def _lift_add(C, F, O, ST: tl.constexpr, SH: tl.constexpr, SW: tl.constexpr,
              TT: tl.constexpr, TH: tl.constexpr, TW: tl.constexpr,
              D: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)*BLOCK+tl.arange(0, BLOCK)
    channel = i % D
    token = i // D
    w = token % TW
    h = token // TW % TH
    t = token // (TW*TH) % TT
    batch = token // (TW*TH*TT)
    parent = ((t//(TT//ST))*SH+h//(TH//SH))*SW+w//(TW//SW)
    ci = (batch*(ST*SH*SW)+parent)*D+channel
    c = tl.load(C+ci, i<TOTAL, other=0).to(tl.float32)
    f = tl.load(F+i, i<TOTAL, other=0).to(tl.float32)
    tl.store(O+i, c+f, i<TOTAL)


def fused_lift_add(coarse, fresh, source, target):
    coarse, fresh = coarse.contiguous(), fresh.contiguous()
    if coarse.shape[0] != fresh.shape[0] or coarse.shape[-1] != fresh.shape[-1]:
        raise ValueError("Batch/channel mismatch")
    out = torch.empty_like(fresh)
    _lift_add[(triton.cdiv(out.numel(), 1024),)](coarse, fresh, out,
        *source, *target, out.shape[-1], out.numel(), 1024)
    return out
