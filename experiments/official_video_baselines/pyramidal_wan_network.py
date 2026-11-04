"""Native Wan modules with the PyramidalWan Appendix A3 token packing.

All projections/norms/FFNs/position encodings are the installed upstream Wan
modules. Only first-frame conditioning and isolated packed attention are new.
The dense backend is a reference; the paper backend uses actual FlexAttention.
"""
from functools import lru_cache
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@lru_cache(maxsize=1)
def _flex():
    from torch.nn.attention.flex_attention import flex_attention
    return torch.compile(flex_attention, dynamic=False, fullgraph=True)


def masks(lengths, text_lengths, device, backend):
    qids = torch.cat([torch.full((n,), i, device=device, dtype=torch.int32)
                      for i, n in enumerate(lengths)])
    kids = torch.cat([torch.full((n,), i, device=device, dtype=torch.int32)
                      for i, n in enumerate(text_lengths)])
    if backend == 'dense':
        return qids[:, None] == qids[None], qids[:, None] == kids[None]
    if backend != 'flex':
        raise ValueError('Packed attention backend must be dense or flex')
    from torch.nn.attention.flex_attention import create_block_mask
    def self_mask(b, h, q, k):
        return qids[q] == qids[k]
    def cross_mask(b, h, q, k):
        return qids[q] == kids[k]
    # The helper pads block indices. Clamp indexing, then reject padding, so
    # nonmultiples of the block size cannot read outside the ownership arrays.
    def bounded_self(b, h, q, k):
        return (q < len(qids)) & (k < len(qids)) & (qids[q.clamp_max(len(qids)-1)] == qids[k.clamp_max(len(qids)-1)])
    def bounded_cross(b, h, q, k):
        return (q < len(qids)) & (k < len(kids)) & (qids[q.clamp_max(len(qids)-1)] == kids[k.clamp_max(len(kids)-1)])
    return (create_block_mask(bounded_self, None, None, len(qids), len(qids), device=str(device)),
            create_block_mask(bounded_cross, None, None, len(qids), len(kids), device=str(device)))


def attention(attn, hidden, context, rotary, mask, backend):
    from diffusers.models.transformers.transformer_wan import _get_qkv_projections
    if attn.add_k_proj is not None:
        raise ValueError('The paper uses Wan2.1 T2V, without I2V image tokens')
    q, k, v = _get_qkv_projections(attn, hidden, context)
    q, k = attn.norm_q(q), attn.norm_k(k)
    q, k, v = [x.unflatten(2, (attn.heads, -1)) for x in (q, k, v)]
    if rotary is not None:
        cos, sin = rotary
        def rotate(x):
            x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
            y = torch.empty_like(x)
            y[..., 0::2] = x1 * cos[..., 0::2] - x2 * sin[..., 1::2]
            y[..., 1::2] = x1 * sin[..., 1::2] + x2 * cos[..., 0::2]
            return y
        q, k = rotate(q), rotate(k)
    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    if backend == 'flex':
        options = None
        if q.dtype == torch.float32:
            # Blackwell SM120 has 99 KiB shared memory; the default FP32
            # head-128 tile exceeds that hardware limit.
            options = {'fwd_BLOCK_M':32, 'fwd_BLOCK_N':32,
                       'bwd_BLOCK_M1':32, 'bwd_BLOCK_N1':32,
                       'bwd_BLOCK_M2':32, 'bwd_BLOCK_N2':32}
        y = _flex()(q, k, v, block_mask=mask, kernel_options=options)
    else:
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[None, None])
    y = y.transpose(1, 2).flatten(2).type_as(q)
    return attn.to_out[1](attn.to_out[0](y))


def packed_block(block, hidden, context, tproj, rotary, self_mask, cross_mask, backend):
    # Same upstream WanTransformerBlock equations; per-token timestep
    # modulation lets differently sized examples share one packed sequence.
    shift, scale, gate, fshift, fscale, fgate = (
        block.scale_shift_table.unsqueeze(0) + tproj.float()).unbind(dim=2)
    norm = (block.norm1(hidden.float()) * (1 + scale) + shift).type_as(hidden)
    out = attention(block.attn1, norm, None, rotary, self_mask, backend)
    hidden = (hidden.float() + out * gate).type_as(hidden)
    norm = block.norm2(hidden.float()).type_as(hidden)
    hidden = hidden + attention(block.attn2, norm, context, None, cross_mask, backend)
    norm = (block.norm3(hidden.float()) * (1 + fscale) + fshift).type_as(hidden)
    out = block.ffn(norm)
    return (hidden.float() + out.float() * fgate).type_as(hidden)


class PaperWanTransformer(nn.Module):
    """Adds one learned vector to every first-frame patch token, all stages.

    Zero initialization is explicit because the paper does not specify it.
    The reference forward calls upstream Wan.forward itself, without packing.
    """
    def __init__(self, dit, *, backend='flex', first_frame_embedding=True):
        super().__init__()
        self.dit = dit
        self.backend = backend
        self.gradient_checkpointing = True
        self.first_frame_embedding = nn.Parameter(torch.zeros(dit.patch_embedding.out_channels),
                                                   requires_grad=first_frame_embedding)
        self.has_first_frame_embedding = first_frame_embedding
        self._mask_cache = {}
        if first_frame_embedding:
            self._hook = dit.patch_embedding.register_forward_hook(self._first_frame)

    def _first_frame(self, module, args, value):
        first = value[:, :, :1] + self.first_frame_embedding[None, :, None, None, None].to(value.dtype)
        return torch.cat((first, value[:, :, 1:]), dim=2)

    def reference(self, inputs, natural, text):
        return [self.dit(x, timestep=z.reshape(1)*1000, encoder_hidden_states=c,
                         return_dict=False)[0] for x, z, c in zip(inputs, natural, text)]

    def forward(self, inputs, natural, text):
        if len(inputs) != len(natural) or len(text) != len(inputs) or not inputs:
            raise ValueError('Packed samples, times and captions must correspond')
        if any(x.shape[0] != 1 for x in inputs) or tuple(self.dit.config.patch_size) != (1, 2, 2):
            raise ValueError('Use one Wan2.1 (1,2,2) sample per packed segment')
        d = self.dit
        tokens, lengths, times, rope_cos, rope_sin = [], [], [], [], []
        grids = []
        for x, z in zip(inputs, natural):
            grid = (x.shape[2], x.shape[3]//2, x.shape[4]//2)
            grids.append(grid)
            cos, sin = d.rope(x)
            rope_cos.append(cos); rope_sin.append(sin)
            token = d.patch_embedding(x).flatten(2).transpose(1, 2).contiguous()
            lengths.append(token.shape[1]); tokens.append(token)
            times.append(z.reshape(1).expand(token.shape[1])*1000)
        hidden = torch.cat(tokens, dim=1)
        z = torch.cat(times)
        context = torch.cat(text, dim=1)
        temb, proj, context, _ = d.condition_embedder(z, context, timestep_seq_len=len(z))
        proj = proj.unflatten(2, (6, -1))
        rotary = (torch.cat(rope_cos, dim=1), torch.cat(rope_sin, dim=1))
        text_lengths = tuple(c.shape[1] for c in text)
        key = (tuple(lengths), text_lengths, str(hidden.device), self.backend)
        if key not in self._mask_cache:
            self._mask_cache[key] = masks(lengths, text_lengths, hidden.device, self.backend)
        self_mask, cross_mask = self._mask_cache[key]
        for block in d.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                def run(h, c, p, *, module=block):
                    return packed_block(module, h, c, p, rotary, self_mask, cross_mask, self.backend)
                hidden = checkpoint(run, hidden, context, proj, use_reentrant=False)
            else:
                hidden = packed_block(block, hidden, context, proj, rotary, self_mask, cross_mask, self.backend)
        shift, scale = (d.scale_shift_table.unsqueeze(0) + temb.unsqueeze(2)).unbind(dim=2)
        hidden = (d.norm_out(hidden.float())*(1+scale)+shift).type_as(hidden)
        hidden = d.proj_out(hidden)
        predictions = []
        for token, (t, h, w) in zip(hidden.split(lengths, dim=1), grids):
            y = token.reshape(1, t, h, w, 1, 2, 2, -1).permute(0, 7, 1, 4, 2, 5, 3, 6)
            predictions.append(y.flatten(6, 7).flatten(4, 5).flatten(2, 3))
        return predictions
