"""Inference pilot: transport inactive FFN neurons, refresh active fine neurons.

All self/cross attention remains native. This is an approximation, not an
equivalent kernel optimization, and does not establish post-training quality.
No upstream module, parameter, or checkpoint is changed.
"""
from __future__ import annotations

import math
import time
import types
import torch
import torch.nn.functional as F


def lift_reference(coarse, source, target):
    if coarse.shape[1] != math.prod(source):
        raise ValueError("Cache/grid mismatch")
    if any(t % s for s, t in zip(source, target)):
        raise ValueError("Only integer nested grids are supported")
    x = coarse.reshape(coarse.shape[0], *source, coarse.shape[-1])
    for axis, (s, t) in enumerate(zip(source, target), 1):
        x = x.repeat_interleave(t // s, dim=axis)
    return x.reshape(coarse.shape[0], math.prod(target), coarse.shape[-1])


def lift_add(coarse, fresh, source, target, backend="triton"):
    if fresh.shape[1] != math.prod(target):
        raise ValueError("Fine/grid mismatch")
    if any(t % s for s, t in zip(source, target)):
        raise ValueError("Only integer nested grids are supported")
    if backend == "torch" or not fresh.is_cuda:
        return lift_reference(coarse, source, target) + fresh
    from .cross_scale_neuron_refresh_triton import fused_lift_add
    return fused_lift_add(coarse, fresh, source, target)


class CrossScaleNeuronRefresh:
    """Cache Y_inactive=FFN(Xc)-Wout[S] GELU(Win[S] Xc+bin[S]).

    On a selected fine call: U(Y_inactive)+Wout[S] GELU(Win[S] Xf+bin[S]).
    The output bias is included exactly once in Y_inactive. The original
    timestep modulation/gates and every attention branch remain untouched.
    Selection is static by weight norm, not a costly runtime neuron router.
    """
    attention_mode = "vsa"

    def __init__(self, model):
        self.model = model
        self.modules = [getattr(b, "ffn", None) or b.mlp for b in model.blocks]
        self.originals = [m.forward for m in self.modules]
        self.stage = 0
        self.grid = None
        self.cache = {}
        self.packed = {}
        self.layers = set()
        self.stages = set()
        self.backend = "triton"
        self.audit = {}
        for layer, module in enumerate(self.modules):
            def forward(module, x, layer=layer):
                return self.forward(layer, module, x)
            module.forward = types.MethodType(forward, module)

    def configure(self, fraction=1., layers=(), stages=(2,), backend="triton"):
        if not 0 < fraction <= 1:
            raise ValueError("fresh fraction must be in (0,1]")
        if backend not in {"torch", "triton"}:
            raise ValueError(backend)
        self.layers = set(layers) if fraction < 1 else set()
        if any(l < 0 or l >= len(self.originals) for l in self.layers):
            raise ValueError("Invalid layer")
        self.stages = set(stages)
        if not self.stages <= {1, 2}:
            raise ValueError("Only middle/fine stages are supported")
        self.backend = backend
        self.cache.clear()
        self.packed.clear()
        start = time.perf_counter()
        with torch.no_grad():
            for l in sorted(self.layers):
                m = self.modules[l]
                fc_in = getattr(m, "fc_in", None) or m.fc1
                fc_out = getattr(m, "fc_out", None) or m.fc2
                wi, wo = fc_in.weight, fc_out.weight
                n = max(64, math.ceil(wi.shape[0]*fraction/64)*64)
                n = min(n, wi.shape[0])
                score = wi.float().norm(dim=1)*wo.float().norm(dim=0)
                indices = score.topk(n).indices.sort().values
                self.packed[l] = (wi.index_select(0, indices).contiguous(),
                    fc_in.bias.index_select(0, indices).contiguous() if fc_in.bias is not None else None,
                    wo.index_select(1, indices).contiguous())
        device = next(self.model.parameters()).device
        if self.layers and device.type == "cuda":
            torch.cuda.synchronize(device)
        self.setup_seconds = time.perf_counter()-start

    def begin_model_forward(self, stage, shape):
        if stage == 0:
            self.cache.clear()
            self.audit = dict(captures=0, refreshes=0, full_ffn=0)
        self.stage = stage
        patch = self.model.patch_size
        patch = (1, patch, patch) if isinstance(patch, int) else tuple(patch)
        if any(s % p for s, p in zip(shape, patch)):
            raise ValueError("Latent shape not patch-aligned")
        self.grid = tuple(s//p for s, p in zip(shape, patch))

    def active(self, module, x, layer):
        wi, bi, wo = self.packed[layer]
        return F.linear(module.act(F.linear(x, wi, bi)), wo)

    def forward(self, layer, module, x):
        if layer not in self.layers:
            return self.originals[layer](x)
        if torch.is_grad_enabled():
            raise RuntimeError("Inference-only pilot; packed weights are not a training implementation")
        if x.ndim != 3 or x.shape[1] != math.prod(self.grid):
            raise ValueError("Pilot requires unsharded [B,T*H*W,C] tokens")
        if self.stage in self.stages:
            if layer not in self.cache:
                raise RuntimeError("No preceding lower-resolution cache")
            source, inactive = self.cache[layer]
            if source == self.grid:
                raise ValueError("Refuse to label same-resolution timestep reuse cross-scale")
            fresh = self.active(module, x, layer)
            y = lift_add(inactive, fresh, source, self.grid, self.backend)
            self.audit["refreshes"] += 1
            # Cache remains in its original coarse grid for a cascade: no
            # repeated upsample nor expanded cache between stages.
            if self.stage == max(self.stages):
                del self.cache[layer]
            return y
        y = self.originals[layer](x)
        self.audit["full_ffn"] += 1
        if self.stage+1 in self.stages:
            # Extra coarse computation, allocation and subtraction are inside
            # the trajectory timer. Do not hide this cost as offline setup.
            self.cache[layer] = (self.grid, (y-self.active(module, x, layer)).detach())
            self.audit["captures"] += 1
        return y

    def restore(self):
        for module, forward in zip(self.modules, self.originals):
            module.forward = forward
        self.cache.clear()
        self.packed.clear()
