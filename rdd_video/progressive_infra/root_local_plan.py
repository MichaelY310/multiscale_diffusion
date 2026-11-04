"""CPU-only shape and indexing contracts shared by GPU runner and tests."""
from dataclasses import dataclass
import math
import torch
from root_local_attention_reference import spatial_tiles,bank_bias

KERNEL_CONFIGS=((64,64,4,3),(128,64,8,3),(128,64,4,3),(64,64,4,5))


@dataclass(frozen=True)
class AttentionPlan:
    shape:tuple
    window:int
    root_slots:int
    heads:int=12
    head_dim:int=128
    density:float=.2

    def __post_init__(self):
        if len(self.shape)!=3 or any(not isinstance(x,int) or x<1 for x in self.shape):
            raise ValueError('Positive T/Hpatch/Wpatch required')
        t,h,w=self.shape
        if not isinstance(self.window,int) or self.window<1 or h%self.window or w%self.window:
            raise ValueError('Equal full-time spatial windows require divisible H/W')
        if t%4 or h%4 or w%4:
            raise ValueError('This benchmark requires non-padded original64 cubes')
        if self.root_slots not in (self.tokens//16,self.tokens//64):
            raise ValueError('Root slots must equal N/16 or N/64 in this pilot')
        if self.heads<1 or self.head_dim not in (64,128) or not math.isfinite(self.density) or not 0<self.density<=1:
            raise ValueError('Invalid heads/dtype geometry or density')

    @property
    def tokens(self):return math.prod(self.shape)
    @property
    def local(self):return self.shape[0]*self.window**2
    @property
    def blocks(self):return self.tokens//64
    @property
    def keep(self):return math.ceil(self.density*self.blocks)
    @property
    def bias(self):return bank_bias(self.root_slots,self.local,.2)

    def window_map(self):
        ids=torch.cat(spatial_tiles(self.shape,self.window,self.window))
        if not torch.equal(ids.sort().values,torch.arange(self.tokens)):
            raise ValueError('Window map must be a permutation')
        return ids

    def cube_map(self):
        t,h,w=self.shape
        grid=torch.arange(self.tokens).reshape(t,h,w)
        return torch.cat([grid[z:z+4,y:y+4,x:x+4].reshape(-1)
                          for z in range(0,t,4) for y in range(0,h,4) for x in range(0,w,4)])


def spatial_pool_projected(x,shape):
    """Current K/V strong control: 4x4 spatial mean, channels unchanged.

    Mean accumulates FP32 and returns input dtype; no extra K/V projection.
    Input is physical flattened BHND, not VSA cube order.
    """
    if x.ndim!=4 or math.prod(shape)!=x.shape[2] or shape[1]%4 or shape[2]%4:
        raise ValueError('BHND with physical T/H/W and H/W divisible4 required')
    b,heads,n,d=x.shape;t,h,w=shape
    return x.float().reshape(b,heads,t,h//4,4,w//4,4,d).mean((4,6)).to(x.dtype).reshape(b,heads,n//16,d).contiguous()
