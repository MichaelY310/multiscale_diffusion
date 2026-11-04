"""Opt-in one/two-span reader. Old checkpoints and model code are untouched.

Same full-channel reader computation as ConditionalPrefixStudent, but separate
reader modules do NOT own duplicate backbones. Exactly18/24 fine blocks run.
The split layout pays for TWO readers; that cost must be counted in benchmarks.
"""
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from csra_layer_layouts import LAYOUTS, layout
from csra_prefix_probe_math import coarse_lift

FORMAT='csra_multispan_conditional_v1'


class SpanReader(nn.Module):
    def __init__(self,h,heads,initializer,coarse_source='aligned'):
        super().__init__()
        if initializer['weight'].shape != (2*h,h) or h%heads or coarse_source not in ('aligned','zero'):
            raise ValueError('Expected full-channel fine-only initializer and valid source')
        self.coarse_source,self.heads=coarse_source,heads
        self.proj=nn.Linear(3*h,h,dtype=torch.float32)
        w=initializer['weight'].double()/initializer['scale'].double()[:,None]
        bias=initializer['output_mean'].double()-initializer['mean'].double()@w
        with torch.no_grad():
            self.proj.weight.zero_(); self.proj.weight[:,:2*h].copy_(w.T.float()); self.proj.bias.copy_(bias.float())
        self.q_norm=nn.LayerNorm(h,elementwise_affine=False,eps=1e-6)
        self.m_norm=nn.LayerNorm(h,elementwise_affine=False,eps=1e-6)
        self.condition=nn.Sequential(nn.SiLU(),nn.Linear(h,4*h))
        self.query=nn.Linear(h,h); self.key_value=nn.Linear(h,2*h); self.read_out=nn.Linear(h,h)
        self.nonlinear=nn.Sequential(nn.Linear(h,2*h),nn.SiLU(),nn.Linear(2*h,h))
        for module in (self.condition[-1],self.read_out,self.nonlinear[-1]):
            nn.init.zeros_(module.weight); nn.init.zeros_(module.bias)

    def forward(self,h0,c,source):
        b,n,h=h0.shape
        if self.coarse_source=='zero':
            memory=torch.zeros((b,n//4,h),device=h0.device,dtype=h0.dtype)
        else:
            memory=source.to(h0.dtype)
        if memory.shape!=(b,n//4,h): raise ValueError('Expected actual native16x16 hidden')
        with torch.autocast(h0.device.type,enabled=False):
            lifted=torch.zeros_like(h0,dtype=torch.float32) if self.coarse_source=='zero' else coarse_lift(source.float(),'coarse_aligned')
            joined=torch.cat((h0.float(),c[:,None].expand(-1,n,-1).float(),lifted),-1)
            base=h0.float()+self.proj(joined)
        with torch.autocast(h0.device.type,dtype=torch.bfloat16,enabled=h0.is_cuda):
            sq,kq,sm,km=self.condition(c).chunk(4,-1)
            qi=self.q_norm(h0)*(1+kq[:,None])+sq[:,None]
            mi=self.m_norm(memory)*(1+km[:,None])+sm[:,None]
            q=self.query(qi).reshape(b,n,self.heads,h//self.heads).transpose(1,2)
            kv=self.key_value(mi).reshape(b,n//4,2,self.heads,h//self.heads)
            k,v=kv.unbind(2)
            a=F.scaled_dot_product_attention(q,k.transpose(1,2),v.transpose(1,2),dropout_p=0.)
            read=self.read_out(a.transpose(1,2).reshape(b,n,h))
            local=self.nonlinear(qi)
        return base+read.float()+local.float()


class LayoutStudent(nn.Module):
    def __init__(self,backbone,initializer,name,coarse_source='aligned',checkpoint_blocks=True):
        super().__init__()
        self.spec=layout(name)
        if (initializer.get('version')!='csra_six_layer_layout_stream_probe_v1'
            or initializer.get('layout')!=self.spec or initializer.get('source_layer')!=13
            or initializer.get('smoke') is not False):
            raise ValueError('Initializer layout/source mismatch or smoke data used for training')
        self.backbone=backbone.float().eval().requires_grad_(True)
        self.coarse_source=coarse_source; self.source_layer=13
        self.spans=LAYOUTS[name]; self.checkpoint_blocks=checkpoint_blocks
        h=backbone.x_embedder.proj.out_channels; heads=backbone.blocks[0].attn.num_heads
        self.readers=nn.ModuleList([SpanReader(h,heads,initializer['fits'][str(n)]['fine_only'],coarse_source)
            for n in range(len(self.spans))])
        for i in self.spec['skipped_layers']: self.backbone.blocks[i].requires_grad_(False)

    def forward(self,batch):
        x,t,y=batch['x'],batch['t'],batch['labels']; m=self.backbone
        gh,gw=x.shape[-2]//m.patch_size,x.shape[-1]//m.patch_size
        starts={s:(n,e) for n,(s,e) in enumerate(self.spans)}
        source=None if self.coarse_source=='zero' else batch['coarse'][13]
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
            h=m.x_embedder(x); c=m.t_embedder(t)+m.y_embedder(y,False)
            cos,sin=m.get_rope(gh,gw,x.device)
            i=0
            while i<len(m.blocks):
                if i in starts:
                    n,end=starts[i]
                    # Recompute the bridge as well as blocks in backward to
                    # keep64 samples/card feasible beside existing training.
                    if self.checkpoint_blocks and torch.is_grad_enabled():
                        h=checkpoint(self.readers[n],h,c,source,use_reentrant=False).to(h.dtype)
                    else: h=self.readers[n](h,c,source).to(h.dtype)
                    i=end
                else:
                    h=checkpoint(m.blocks[i],h,c,cos,sin,use_reentrant=False) if self.checkpoint_blocks and torch.is_grad_enabled() else m.blocks[i](h,c,cos,sin)
                    i+=1
            return m.unpatchify(m.final_layer(h,c),gh,gw)
