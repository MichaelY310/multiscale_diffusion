"""Opt-in trainable coarse-dependent prefix replacement. Old models unchanged."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from csra_prefix_probe_math import coarse_lift


def check_prefix_initializer(payload, prefix_depth, source_layer=13):
    if (payload.get('version') != 'prefix_information_v1'
        or payload.get('prefix_depth',14) != prefix_depth
        or payload.get('source_layer',13) != source_layer):
        raise ValueError('Ridge target/source depth does not match the requested student')


class PrefixLiftStudent(nn.Module):
    def __init__(self, backbone, initializer, coarse_source='aligned', prefix_depth=14,
                 source_layer=13, checkpoint_blocks=True):
        super().__init__()
        if coarse_source not in ('aligned','zero') or not 0 < prefix_depth < len(backbone.blocks):
            raise ValueError('Invalid source or prefix length')
        self.backbone = backbone.float().eval().requires_grad_(True)
        self.coarse_source = coarse_source
        self.prefix_depth, self.source_layer = prefix_depth, source_layer
        self.checkpoint_blocks = checkpoint_blocks
        h = backbone.x_embedder.proj.out_channels
        if initializer['weight'].shape != (2*h,h):
            raise ValueError('Both arms must initialize from the SAME fine-only ridge')
        self.proj = nn.Linear(3*h,h,dtype=torch.float32)
        w = initializer['weight'].double()/initializer['scale'].double()[:,None]
        bias = initializer['output_mean'].double()-initializer['mean'].double()@w
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.weight[:,:2*h].copy_(w.T.float())
            self.proj.bias.copy_(bias.float())
        for block in self.backbone.blocks[:prefix_depth]:
            block.requires_grad_(False)

    def lift(self,h0,c,source):
        # Full-rank, no low-rank output bottleneck. This reference deliberately
        # concatenates lifted source; optimize/fuse only after quality works.
        with torch.autocast(h0.device.type,enabled=False):
            if self.coarse_source == 'zero':
                source = torch.zeros_like(h0, dtype=torch.float32)
            else:
                source = coarse_lift(source.float(),'coarse_aligned')
            joined = torch.cat((h0.float(),c[:,None].expand(-1,h0.shape[1],-1).float(),source),-1)
            return h0.float()+self.proj(joined)

    def forward(self,batch,return_hidden=False):
        x,t,y = batch['x'],batch['t'],batch['labels']
        m = self.backbone
        gh,gw = x.shape[-2]//m.patch_size,x.shape[-1]//m.patch_size
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
            h0 = m.x_embedder(x)
            c = m.t_embedder(t)+m.y_embedder(y,False)
            source = None if self.coarse_source=='zero' else batch['coarse'][self.source_layer]
            h = self.lift(h0,c,source).to(h0.dtype)
            bridge_hidden = h
            cos,sin = m.get_rope(gh,gw,x.device)
            for block in m.blocks[self.prefix_depth:]:
                if self.checkpoint_blocks and torch.is_grad_enabled():
                    h = checkpoint(block,h,c,cos,sin,use_reentrant=False)
                else:
                    h = block(h,c,cos,sin)
            output = m.unpatchify(m.final_layer(h,c),gh,gw)
            # Opt-in training diagnostic; default tensor output and state_dict
            # remain unchanged. Exposes the actual rounded suffix input.
            return (output,bridge_hidden) if return_hidden else output


class ConditionalPrefixStudent(PrefixLiftStudent):
    """Opt-in nonlinear prefix bridge; native coarse K/V, current fine queries.

Starts EXACTLY at the same fine-only affine function as PrefixLiftStudent.
The added full-channel attention and pointwise FFN are not claimed as novel
modules. The experiment tests their placement before the skipped fine prefix.
No fine-to-fine self attention is added. Source stays native (N/4 tokens).
"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        h = self.proj.out_features
        self.heads = self.backbone.blocks[0].attn.num_heads
        if h % self.heads:
            raise ValueError('Hidden width must divide heads')
        self.q_norm = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.m_norm = nn.LayerNorm(h, elementwise_affine=False, eps=1e-6)
        self.condition = nn.Sequential(nn.SiLU(), nn.Linear(h, 4*h))
        self.query = nn.Linear(h, h)
        self.key_value = nn.Linear(h, 2*h)
        self.read_out = nn.Linear(h, h)
        self.nonlinear = nn.Sequential(nn.Linear(h, 2*h), nn.SiLU(), nn.Linear(2*h, h))
        # Common initial output across linear/nonlinear and aligned/zero arms.
        for layer in (self.condition[-1], self.read_out, self.nonlinear[-1]):
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def lift(self, h0, c, source):
        base = super().lift(h0, c, source)
        b, n, h = h0.shape
        if self.coarse_source == 'zero':
            memory = torch.zeros((b, n//4, h), device=h0.device, dtype=h0.dtype)
        else:
            memory = source.to(h0.dtype)
        if memory.shape != (b, n//4, h):
            raise ValueError('Expected native coarse grid with one quarter fine tokens')
        # Re-enable the same BF16 AMP used by the backbone. Original affine base
        # is FP32, and zero residual paths preserve the original initial function.
        with torch.autocast(h0.device.type, dtype=torch.bfloat16, enabled=h0.is_cuda):
            shift_q, scale_q, shift_m, scale_m = self.condition(c).chunk(4, -1)
            q_input = self.q_norm(h0)*(1+scale_q[:, None])+shift_q[:, None]
            m_input = self.m_norm(memory)*(1+scale_m[:, None])+shift_m[:, None]
            q = self.query(q_input).reshape(b, n, self.heads, h//self.heads).transpose(1, 2)
            kv = self.key_value(m_input).reshape(b, n//4, 2, self.heads, h//self.heads)
            k, v = kv.unbind(2)
            attended = F.scaled_dot_product_attention(q, k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.)
            attended = attended.transpose(1, 2).reshape(b, n, h)
            read = self.read_out(attended)
            local = self.nonlinear(q_input)
        return base + read.float() + local.float()


def prefix_student_class(kind):
    if kind == 'linear': return PrefixLiftStudent
    if kind == 'conditional_attention': return ConditionalPrefixStudent
    raise ValueError(f'Unknown prefix bridge: {kind}')


def prefix_checkpoint_format(kind):
    return 'csra_prefix_linear_v7' if kind == 'linear' else 'csra_prefix_conditional_v8'
