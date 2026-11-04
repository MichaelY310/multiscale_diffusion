"""No adapter: one trainable RDD backbone produces AND consumes coarse updates.

The saved source inputs are teacher-trajectory states, but their block updates
are recomputed by the current student. No detached/frozen feature bank is used
in training. Fine h <- fine h + nearest(coarse h_after - coarse h_before).
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

MASKS = {'dense': (), 'reuse6': (3, 6, 10, 13, 17, 20),
         'reuse8': (2, 5, 8, 10, 13, 15, 18, 21)}


def lift(delta, gh, gw):
    b, n, c = delta.shape
    if n == gh*gw:
        return delta
    g = int(n ** .5)
    if g*g != n:
        raise ValueError('Square 2D source required')
    return F.interpolate(delta.transpose(1, 2).reshape(b, c, g, g),
                         size=(gh, gw), mode='nearest').flatten(2).transpose(1, 2)


class DirectReuseStudent(nn.Module):
    def __init__(self, backbone, layers, checkpoint_blocks=True):
        super().__init__()
        self.backbone = backbone
        self.layers = tuple(layers)
        if len(set(layers)) != len(layers) or any(l < 0 or l >= len(backbone.blocks) for l in layers):
            raise ValueError('Invalid layer mask')
        if getattr(backbone, 'd_prediction', False):
            raise ValueError('Pilot only supports original epsilon/variance head')
        self.checkpoint_blocks = checkpoint_blocks

    def stage(self, x, t, y, cache=None, capture=False):
        m = self.backbone
        gh, gw = x.shape[-2] // m.patch_size, x.shape[-1] // m.patch_size
        captured = {}
        with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=x.is_cuda):
            h = m.x_embedder(x)
            cos, sin = m.get_rope(gh, gw, x.device)
            # Fixed conditional corpus: no independent classifier-free dropout.
            condition = m.t_embedder(t) + m.y_embedder(y, False)
            for i, block in enumerate(m.blocks):
                before = h
                if cache is not None and i in self.layers:
                    h = h + lift(cache[i], gh, gw).to(h.dtype)
                elif self.checkpoint_blocks and torch.is_grad_enabled() and h.requires_grad:
                    h = checkpoint(block, h, condition, cos, sin, use_reentrant=False)
                else:
                    h = block(h, condition, cos, sin)
                if capture and i in self.layers:
                    captured[i] = h - before  # Intentionally NO detach.
            return m.unpatchify(m.final_layer(h, condition), gh, gw), captured

    def forward(self, batch):
        middle, cache = self.stage(batch['source16_x'], batch['source16_t'],
                                   batch['labels'], capture=True)
        fine, _ = self.stage(batch['x'], batch['t'], batch['labels'], cache=cache)
        coarse, _ = self.stage(batch['source8_x'], batch['source8_t'], batch['labels'])
        return fine, middle, coarse


class DirectReuseRollout:
    """Same updated model at ALL resolutions; cache last t=300 middle NFE."""
    def __init__(self, student):
        self.student = student
        self.cache = None
        self.labels = None
        self.source_calls = 0
        self.replaced_blocks = 0

    def forward(self, x, t, y):
        self.labels = y.detach().cpu()
        source = x.shape[-1] == 16 and bool(torch.all(t == 300))
        if source:
            out, cache = self.student.stage(x, t, y, capture=True)
            # Inference cache is prepared ONCE, outside all later fine blocks.
            self.cache = {l: lift(d, 16, 16).detach() for l, d in cache.items()}
            self.source_calls += 1
            return out
        if x.shape[-1] == 32 and self.student.layers:
            if self.cache is None:
                raise RuntimeError('No middle boundary cache: incompatible schedule')
            self.replaced_blocks += len(self.student.layers)
            return self.student.stage(x, t, y, cache=self.cache)[0]
        return self.student.stage(x, t, y)[0]
