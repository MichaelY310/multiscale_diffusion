"""CPU oracle for a proposed anchor/detail FFN epilogue, not an NVFP4 GEMM.

Logical row-major scales are deliberately NOT advertised as CUTLASS-ready.
Whole-model use requires trained per-layer anchors and new normalization.
"""
from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass
class PackedNVFP4:
    data: torch.Tensor             # [rows, padded_K/2], uint8, low nibble first
    scales: torch.Tensor           # [rows, padded_K/16], E4M3 bytes
    global_scale: float
    width: int
    clipped: torch.Tensor          # counts per 16-value block


def e2m1_codes(x):
    """Finite inputs, saturated round-to-nearest-even, preserves signed zero."""
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Non-finite input is not a valid inference activation')
    table = x.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    # Saturate BEFORE distances: at huge finite magnitudes all eight FP32
    # distances otherwise collapse to the same value and spuriously choose0.
    distance = (x.abs().clamp(max=6).unsqueeze(-1) - table).abs()
    nearest = distance == distance.min(-1, keepdim=True).values
    # Binary code parity corresponds to the last mantissa bit at midpoint ties.
    preference = torch.tensor([0, 8, 2, 10, 4, 12, 6, 14], device=x.device)
    score = torch.where(nearest, preference, 99)
    index = score.argmin(-1).to(torch.uint8)
    return index | (torch.signbit(x).to(torch.uint8) << 3)


def decode_e2m1(code):
    levels = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=code.device)
    return levels[(code & 7).long()] * torch.where((code & 8) != 0, -1., 1.)


def pack_nvfp4(x, global_scale=None):
    """Reference quantizer, group16 E4M3 scales. No CUDA is invoked.

    A supplied global scale allows calibration; clipping counts remain visible.
    This is not a stochastic training quantizer or a custom backward operator.
    """
    if x.device.type != 'cpu' or x.ndim != 2 or x.shape[1] == 0:
        raise ValueError('Nonempty CPU [rows,K] required')
    x = x.detach().float()
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Non-finite activation')
    width = x.shape[1]
    padded = math.ceil(width / 16) * 16
    blocks = F.pad(x, (0, padded-width)).reshape(x.shape[0], -1, 16)
    maxima = blocks.abs().amax(-1)
    if global_scale is None:
        # This tensor-wide amax is NOT free: a GPU implementation must charge it.
        global_scale = float(maxima.max()) / (6*448) if maxima.numel() and maxima.max() > 0 else 1.
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError('A finite positive global scale is required')
    global_scale = float(torch.tensor(global_scale, dtype=torch.float32))
    if not math.isfinite(global_scale) or global_scale <= 0:
        raise ValueError('Global scale must be representable as positive FP32')
    raw = (maxima / global_scale) / 6
    # Avoid zero denominator for tiny nonzero blocks; zero blocks use scale0.
    raw = torch.where(maxima == 0, 0., raw.clamp(1/512, 448))
    sf = raw.to(torch.float8_e4m3fn)
    effective = sf.float() * global_scale
    divisor = torch.where(effective == 0, 1., effective)
    normalized = (blocks / divisor.unsqueeze(-1)).clamp(-6, 6)
    # Match producer's canonical +0 in all-zero scale blocks. Signed zero is
    # otherwise preserved when a nonzero scale exists in the same block.
    normalized = torch.where(effective.unsqueeze(-1) == 0, 0., normalized)
    clipped = (blocks.abs() > 6*effective.unsqueeze(-1)).sum(-1)
    codes = e2m1_codes(normalized).reshape(x.shape[0], padded)
    data = codes[:, 0::2] | (codes[:, 1::2] << 4)
    return PackedNVFP4(data, sf.view(torch.uint8), float(global_scale), width, clipped)


def unpack_nvfp4(p):
    codes = torch.stack((p.data & 15, p.data >> 4), -1).flatten(-2)
    values = decode_e2m1(codes).reshape(p.data.shape[0], -1, 16)
    sf = p.scales.view(torch.float8_e4m3fn).float()
    return (values * sf.unsqueeze(-1) * p.global_scale).flatten(-2)[:, :p.width]


def fresh_detail_gelu(parent_pre, detail_pre):
    """Same signs and explicit arithmetic as the CUDA kernel; all four fresh."""
    if parent_pre.ndim != 2 or detail_pre.shape != (*parent_pre.shape[:1], 3, parent_pre.shape[1]):
        raise ValueError('Parent [Nc,G], detail [Nc,3,G] required')
    a = parent_pre.float()
    b, c, d = detail_pre.float().unbind(1)
    y0 = F.gelu(((a+b)+c)+d)
    y1 = F.gelu(((a-b)+c)-d)
    y2 = F.gelu(((a+b)-c)-d)
    y3 = F.gelu(((a-b)-c)+d)
    return torch.stack(((((y0-y1)+y2)-y3)/4,
                        (((y0+y1)-y2)-y3)/4,
                        (((y0-y1)-y2)+y3)/4), 1)


def fused_epilogue_reference(parent_pre, detail_pre, global_scale=None):
    high = fresh_detail_gelu(parent_pre, detail_pre)
    return pack_nvfp4(high.flatten(0, 1), global_scale)


def unique_cache_budget(d=1536, g=8960, nc=15360, layers=30):
    """FLOP bound vs a strong ordinary current-mean, high-only FFN control.

    Counts self QKV+out (4D²), text Q+out (2D²), mean/detail FFN.
    Ignores attention, text K/V, norms, VAE, hardware and cache overhead:
    this is an optimistic linear-work bound, NOT a speed prediction.
    """
    removed = d*g                 # one known mean W1 row per parent
    baseline = 4*6*d*d + 7*d*g   # four self/cross rows; four W1 + three W2
    return dict(unique_removed_linear_fraction=removed/baseline,
                parent_preactivation_cache_gib=nc*g*2*layers/2**30,
                anchor_read_bytes_per_layer=nc*g*2,
                gpu_speed_measured=False, trained_quality_measured=False)
