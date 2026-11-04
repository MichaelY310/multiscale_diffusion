"""Common ImageNet comparison blocks, with explicit author position policies.

RDD itself uses native_rdd_unified directly. Other scalar paths call that same
class and adapt only its clock and output interpretation. Product states need
joint token attention, per-band clocks, and (for Haar) channel projections.
Those changes are explicit architecture transplants, not author-model parity.
"""
import torch
from torch import nn
from .registry import build, register
from .comparison_position import ComparisonPosition


class PositionedAttention(nn.Module):
    """Keep common QKV/projection weights; inject only the positional rule."""
    def __init__(self, original, position):
        super().__init__()
        self.qkv, self.proj = original.qkv, original.proj
        self.num_heads, self.head_dim, self.scale = original.num_heads, original.head_dim, original.scale
        self.position = position

    def forward(self, x, position, unused_sin=None):
        n, length, width = x.shape
        q, k, v = self.qkv(x).reshape(n, length, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.position.rotate(q, position), self.position.rotate(k, position)
        weights = ((q @ k.transpose(-2, -1))*self.scale).softmax(-1)
        return self.proj((weights @ v).transpose(1, 2).reshape(n, length, width))


@register('model', 'comparison_dit')
class ComparisonDiT(nn.Module):
    def __init__(self, architecture, time_scale=1000., heads=None,
                 channels=None, product_order='coarse_first', causal_scales=False,
                 position='legacy_rdd'):
        super().__init__()
        self.core = build('model', {'name': 'native_rdd_unified', 'args': architecture})
        self.position = ComparisonPosition(position)
        if position != 'legacy_rdd':
            for block in self.core.blocks:
                block.attn = PositionedAttention(block.attn, self.position)
        self.time_scale = time_scale
        self.heads = heads
        self.channels = list(channels or [self.core.in_channels])
        self.order = product_order
        self.causal = causal_scales
        if product_order not in ('coarse_first', 'fine_first'):
            raise ValueError('Unknown product order')
        if heads and len(heads) * self.core.in_channels != self.core.out_channels:
            raise ValueError('Named heads must cover exactly the native output channels')
        if self.core.d_prediction or self.core.repa_dim is not None:
            raise ValueError('Comparison backbone excludes auxiliary drift/REPA model branches')
        self.inputs, self.outputs = nn.ModuleDict(), nn.ModuleDict()
        # Ordinary Laplacian bands share the exact legacy input/output ports.
        # Haar detail channels require an explicit additional projection.
        for c in sorted(set(self.channels) - {self.core.in_channels}):
            self.inputs[str(c)] = type(self.core.x_embedder)(self.core.patch_size, c, self.core.hidden_size)
            self.outputs[str(c)] = type(self.core.final_layer)(self.core.hidden_size, self.core.patch_size, c)
            nn.init.zeros_(self.outputs[str(c)].adaLN_modulation[-1].weight)
            nn.init.zeros_(self.outputs[str(c)].adaLN_modulation[-1].bias)
            nn.init.zeros_(self.outputs[str(c)].linear.weight)
            nn.init.zeros_(self.outputs[str(c)].linear.bias)

    def _output(self, value):
        if self.heads:
            return dict(zip(self.heads, value.chunk(len(self.heads), 1)))
        # Preserve the native model and its initialization, including its two
        # output slots. Only RDD interprets slot 2 as learned variance.
        return value[:, :self.core.in_channels]

    def _scalar(self, x, t, y):
        if self.position.name == 'legacy_rdd':
            return self.core(x, t, y)
        net = self.core
        h, w = x.shape[-2]//net.patch_size, x.shape[-1]//net.patch_size
        z = net.x_embedder(x)
        additive = self.position.additive(net.hidden_size, h, w, z.device)
        if additive is not None:
            z = z+additive
        rotary = self.position.rotary(net, h, w, z.device)
        cond = net.t_embedder(t)+net.y_embedder(y, net.training)
        for block in net.blocks:
            z = block(z, cond, rotary, None)
        return net.unpatchify(net.final_layer(z, cond), h, w)

    def forward(self, x, t, stage=0, y=None, drop_mask=None, var_scale=None):
        if y is None:
            raise ValueError('ImageNet comparison requires explicit class labels')
        if var_scale is not None:
            raise ValueError('Variance-aware LapFlow is not in this common-backbone contract')
        if torch.is_tensor(x):
            if x.ndim != 4 or t.ndim > 1:
                raise ValueError('Scalar comparison paths require BCHW and scalar/B time')
            if t.ndim == 0:
                t = t.expand(len(x))
            return self._output(self._scalar(x, t*self.time_scale, y))

        is_dict = isinstance(x, dict)
        keys = list(x) if is_dict else list(range(len(x)))
        values = [x[k] for k in keys]
        if not values or self.heads:
            raise ValueError('Product states require nonempty single-output bands')
        n = len(values[0])
        if t.ndim == 0:
            t = t.expand(n)
        if t.shape[0] != n:
            raise ValueError('Clock batch mismatch')
        if len(values) == 1 and values[0].shape[1] == self.core.in_channels:
            clock = t[:, 0] if t.ndim == 2 else t
            out = self._output(self._scalar(values[0], clock*self.time_scale, y))
            return {keys[0]: out} if is_dict else [out]

        net = self.core
        label = net.y_embedder(y, net.training)
        tokens, conds, positions, shapes, ids = [], [], [], [], []
        for j, (key, value) in enumerate(zip(keys, values)):
            c, h, w = value.shape[1:]
            if c not in self.channels or h % net.patch_size or w % net.patch_size:
                raise ValueError('Invalid product-state port/grid')
            embed = net.x_embedder if c == net.in_channels else self.inputs[str(c)]
            z = embed(value)
            gh, gw = h//net.patch_size, w//net.patch_size
            additive = self.position.additive(net.hidden_size, gh, gw, value.device)
            if additive is not None:
                z = z+additive
            positions.append(self.position.rotary(net, gh, gw, value.device))
            clock = t[:, j] if t.ndim == 2 else t
            cond = net.t_embedder(clock*self.time_scale)+label
            tokens.append(z); conds.append(cond[:, None].expand(-1, z.shape[1], -1))
            shapes.append((c, gh, gw, z.shape[1]))
            ids.extend([j]*z.shape[1])
        z, cond = torch.cat(tokens, 1), torch.cat(conds, 1)
        position = self.position.concatenate(positions)
        ids = torch.tensor(ids, device=z.device)
        allowed = torch.ones(len(ids), len(ids), dtype=torch.bool, device=z.device)
        if self.causal:
            allowed &= ids[None, :] <= ids[:, None] if self.order == 'coarse_first' else ids[None, :] >= ids[:, None]
        if drop_mask is not None:
            if len(drop_mask) != len(values) or not bool(torch.as_tensor(drop_mask).any()):
                raise ValueError('Invalid product-state active mask')
            allowed &= torch.as_tensor(drop_mask, device=z.device).bool()[ids][None, :]
        if not bool(allowed.any(-1).all()):
            raise ValueError('An attention query has no active key')
        for block in net.blocks:
            shift, scale, gate, shift2, scale2, gate2 = block.adaLN_modulation(cond).chunk(6, -1)
            h = block.norm1(z)*(1+scale)+shift
            a = block.attn
            q, k, v = a.qkv(h).reshape(n, h.shape[1], 3, a.num_heads, a.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = self.position.rotate(q, position), self.position.rotate(k, position)
            weights = ((q @ k.transpose(-2, -1))*a.scale).masked_fill(~allowed, -torch.inf).softmax(-1)
            h = a.proj((weights @ v).transpose(1, 2).reshape_as(z))
            z = z+gate*h
            z = z+gate2*block.mlp(block.norm2(z)*(1+scale2)+shift2)
        outputs = []
        offset = 0
        for c, gh, gw, length in shapes:
            final = net.final_layer if c == net.in_channels else self.outputs[str(c)]
            part, cc = z[:, offset:offset+length], cond[:, offset:offset+length]
            shift, scale = final.adaLN_modulation(cc).chunk(2, -1)
            part = final.linear(final.norm_final(part)*(1+scale)+shift)
            out_channels = net.out_channels if c == net.in_channels else c
            part = net.unpatchify(part, gh, gw, channels=out_channels)[:, :c]
            outputs.append(part); offset += length
        return dict(zip(keys, outputs)) if is_dict else outputs

    def forward_with_cfg(self, t, x, y, cfg_scale, var_scale=None):
        """LapFlow's time-first CFG API, preserving the joint-band state."""
        out = self.forward(x, t, y=y, var_scale=var_scale)
        result = []
        for value in out:
            conditional, unconditional = value.chunk(2)
            guided = unconditional+cfg_scale*(conditional-unconditional)
            result.append(torch.cat([guided, guided]))
        return result
