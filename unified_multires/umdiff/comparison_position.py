"""Author positional rules on the common image DiT's token grids.

This changes spatial position only. It does not import an author's attention,
QK normalization, patch projection or time embedding. Video-derived policies
use T=1. See README_ZH.md for the explicit token/axis dimension adaptations.
"""
import numpy as np
import torch

from .position import rotary_2d


def sincos_2d(dim, h, w):
    """DFM SiT / LapFlow default: fixed MAE sin-cos, width axis first."""
    if dim % 4:
        raise ValueError('Absolute 2D sin-cos needs hidden_size divisible by four')
    grid = np.stack(np.meshgrid(np.arange(w, dtype=np.float32),
                                np.arange(h, dtype=np.float32)), axis=0)
    parts = []
    for pos in grid:
        omega = np.arange(dim//4, dtype=np.float64)
        omega /= dim/4.
        omega = 1. / 10000**omega
        out = np.einsum('m,d->md', pos.reshape(-1), omega)
        parts.append(np.concatenate([np.sin(out), np.cos(out)], axis=1))
    return torch.from_numpy(np.concatenate(parts, axis=1)).float()


class ComparisonPosition:
    """Stateless positional rule; no weights, RNG draws or checkpoint keys."""
    NAMES = {'legacy_rdd', 'pixelflow_rope', 'dfm_sincos', 'lapflow_sincos',
             'pyramid_flow_rope', 'wan_rope'}

    def __init__(self, name='legacy_rdd'):
        if name not in self.NAMES:
            raise ValueError(f'Unknown comparison positional rule: {name}')
        self.name = name

    def additive(self, dim, h, w, device):
        if self.name in ('dfm_sincos', 'lapflow_sincos'):
            return sincos_2d(dim, h, w).to(device)
        return None

    def rotary(self, core, h, w, device):
        dim = core.head_dim
        if self.name == 'legacy_rdd':
            return core.get_rope(h, w, device)
        if self.name in ('dfm_sincos', 'lapflow_sincos'):
            return None
        if self.name == 'pixelflow_rope':
            return tuple(x.to(device) for x in rotary_2d(dim, ((0, 0), (h, w)), (h, w)))
        if self.name == 'pyramid_flow_rope':
            if dim < 8 or dim % 4:
                raise ValueError('Pyramid Flow positional transplant needs head_dim >=8 divisible by4')
            # Source [16,24,24] at head64 -> [16,28,28] at common head72.
            # T=1 has temporal coordinate0; no autoregressive history tokens.
            dt = max(4, 4*(dim//16))
            dims = [dt, (dim-dt)//2, (dim-dt)//2]
            yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing='ij')
            ids = [torch.zeros(h*w), yy.flatten(), xx.flatten()]
            parts = []
            for pos, d in zip(ids, dims):
                scale = torch.arange(0, d, 2, dtype=torch.float64, device=device)/d
                omega = 1.0/(10000**scale)
                angle = torch.einsum('...n,d->...nd', pos[None].to(device), omega)
                c, s = torch.cos(angle), torch.sin(angle)
                parts.append(torch.stack([c, -s, s, c], -1).view(1, h*w, d//2, 2, 2).float())
            return torch.cat(parts, -3)[0]
        # Wan's per-axis allocation and complex128 rotation, restricted to T=1.
        if dim < 6 or dim % 2:
            raise ValueError('Wan positional transplant needs even head_dim >=6')
        dims = [dim-4*(dim//6), 2*(dim//6), 2*(dim//6)]
        parts = []
        for count, d in zip((1, h, w), dims):
            freqs = torch.outer(torch.arange(count),
                                1.0/torch.pow(10000, torch.arange(0, d, 2).to(torch.float64).div(d)))
            parts.append(torch.polar(torch.ones_like(freqs), freqs))
        return torch.cat([parts[0].view(1, 1, 1, -1).expand(1, h, w, -1),
                          parts[1].view(1, h, 1, -1).expand(1, h, w, -1),
                          parts[2].view(1, 1, w, -1).expand(1, h, w, -1)], -1).reshape(h*w, -1).to(device)

    def concatenate(self, values):
        if values[0] is None:
            return None
        if isinstance(values[0], tuple):
            return tuple(torch.cat([v[i] for v in values]) for i in range(2))
        return torch.cat(values)

    def rotate(self, x, position):
        """x is B,heads,tokens,head_dim. Keep each author's arithmetic."""
        if position is None:
            return x
        if self.name == 'wan_rope':
            pair = torch.view_as_complex(x.to(torch.float64).reshape(*x.shape[:-1], -1, 2))
            return torch.view_as_real(pair*position[None, None]).flatten(-2).float()
        if self.name == 'pyramid_flow_rope':
            pair = x.float().reshape(*x.shape[:-1], -1, 1, 2)
            matrix = position[None, None]
            out = matrix[..., 0]*pair[..., 0]+matrix[..., 1]*pair[..., 1]
            return out.reshape_as(x).type_as(x)
        cos, sin = position
        real, imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)
        rotated = torch.stack([-imag, real], dim=-1).flatten(3)
        if self.name == 'pixelflow_rope':
            return (x.float()*cos[None, None]+rotated.float()*sin[None, None]).to(x.dtype)
        return x*cos[None, None]+rotated*sin[None, None]
