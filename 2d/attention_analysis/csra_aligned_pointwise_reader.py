"""Opt-in exact gate0 execution: retain spatial coarse affine and fine MLP.

No Q/K/V projections or SDPA are executed. This is NOT a new trained checkpoint.
Only use after loading a validated old student, replacing its reader modules.
"""
import torch
from torch import nn
from csra_prefix_probe_math import coarse_lift


class AlignedPointwiseReader(nn.Module):
    def __init__(self,reader):
        super().__init__()
        if reader.coarse_source!='aligned':raise ValueError('This opt-in runtime supports aligned source only')
        self.proj=reader.proj
        self.q_norm=reader.q_norm
        self.condition=reader.condition
        self.nonlinear=reader.nonlinear

    def forward(self,h0,c,source):
        b,n,h=h0.shape
        if source.shape!=(b,n//4,h):raise ValueError('Expected actual native coarse hidden')
        with torch.autocast(h0.device.type,enabled=False):
            lifted=coarse_lift(source.float(),'coarse_aligned')
            joined=torch.cat((h0.float(),c[:,None].expand(-1,n,-1).float(),lifted),-1)
            base=h0.float()+self.proj(joined)
        with torch.autocast(h0.device.type,dtype=torch.bfloat16,enabled=h0.is_cuda):
            sq,aq,_,_=self.condition(c).chunk(4,-1)
            qi=self.q_norm(h0)*(1+aq[:,None])+sq[:,None]
            local=self.nonlinear(qi)
        return base+local.float()
