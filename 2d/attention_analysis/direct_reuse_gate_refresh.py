"""Optional adapter-free variant: reuse branch values, refresh ORIGINAL gates.

No new parameters. Source attention/MLP VALUES remain from coarse; fine original
adaLN gates use the current fine timestep. Not recomputing fine QKV or MLP.
"""
import torch
from torch.utils.checkpoint import checkpoint
from direct_reuse_fullmodel import DirectReuseStudent, lift


def branch_forward(block, h, c, cos, sin):
    shift_a,scale_a,gate_a,shift_m,scale_m,gate_m=block.adaLN_modulation(c).chunk(6,dim=1)
    a=block.attn(block.norm1(h)*(1+scale_a[:,None])+shift_a[:,None],cos,sin)
    h=h+gate_a[:,None]*a
    m=block.mlp(block.norm2(h)*(1+scale_m[:,None])+shift_m[:,None])
    return h+gate_m[:,None]*m,a,m


class GateRefreshStudent(DirectReuseStudent):
    def stage(self,x,t,y,cache=None,capture=False):
        m=self.backbone;gh,gw=x.shape[-2]//m.patch_size,x.shape[-1]//m.patch_size;captured={}
        with torch.autocast(x.device.type,dtype=torch.bfloat16,enabled=x.is_cuda):
            h=m.x_embedder(x);cos,sin=m.get_rope(gh,gw,x.device)
            c=m.t_embedder(t)+m.y_embedder(y,False)
            for i,block in enumerate(m.blocks):
                if cache is not None and i in self.layers:
                    gate_a,gate_m=block.adaLN_modulation(c).chunk(6,dim=1)[2::3]
                    a,v=cache[i]
                    h=h+gate_a[:,None]*lift(a,gh,gw)
                    h=h+gate_m[:,None]*lift(v,gh,gw)
                elif capture and i in self.layers:
                    def run(h,c,cos,sin,block=block):return branch_forward(block,h,c,cos,sin)
                    if self.checkpoint_blocks and torch.is_grad_enabled() and h.requires_grad:
                        h,a,v=checkpoint(run,h,c,cos,sin,use_reentrant=False)
                    else:h,a,v=run(h,c,cos,sin)
                    captured[i]=(a,v)  # no detach during whole-model training
                elif self.checkpoint_blocks and torch.is_grad_enabled() and h.requires_grad:
                    h=checkpoint(block,h,c,cos,sin,use_reentrant=False)
                else:h=block(h,c,cos,sin)
            return m.unpatchify(m.final_layer(h,c),gh,gw),captured


class GateRefreshRollout:
    def __init__(self,student):
        self.student=student;self.cache=None;self.labels=None;self.source_calls=0;self.replaced_blocks=0
    def forward(self,x,t,y):
        self.labels=y.detach().cpu()
        if x.shape[-1]==16 and bool(torch.all(t==300)):
            out,cache=self.student.stage(x,t,y,capture=True)
            self.cache={l:tuple(lift(v,16,16).detach() for v in pair) for l,pair in cache.items()}
            self.source_calls+=1
            return out
        if x.shape[-1]==32 and self.student.layers:
            if self.cache is None:raise RuntimeError('Missing cache')
            self.replaced_blocks+=len(self.student.layers)
            return self.student.stage(x,t,y,cache=self.cache)[0]
        return self.student.stage(x,t,y)[0]
