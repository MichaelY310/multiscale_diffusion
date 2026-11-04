import torch
from .registry import register
from .operators import Projection,block_average,resize,channel_down,channel_up

@register('representation','identity')
class Identity:
    def encode(self,x):return x
    def decode(self,x):return x

@register('representation','average_pool')
class AveragePool:
    """Training input projection; block-mean arithmetic matches RDD source."""
    def __init__(self,factor=2,detach=False):self.factor=factor;self.detach=detach
    def encode(self,x):
        if self.detach:x=x.detach()
        if x.ndim==5:return block_average(x,self.factor)
        b,c,h,w=x.shape;r=self.factor
        if r<1 or h%r or w%r:raise ValueError('Invalid average-pool input grid')
        return x if r==1 else x.reshape(b,c,h//r,r,w//r,r).mean(dim=(3,5))
    def decode(self,x):raise ValueError('Average-pool training projection has no inverse')

@register('representation','laplacian')
class Laplacian:
    def __init__(self,levels=2,down='bilinear',up='bilinear',antialias=False,order='coarse_first'):
        if levels<1 or order not in ('coarse_first','fine_first'):raise ValueError('Invalid pyramid')
        self.levels=levels;self.op=Projection(down=down,up=up,antialias=antialias);self.order=order
    def encode(self,x):
        parts=[]
        for _ in range(self.levels-1):
            low=self.op.down(x);parts.append(x-self.op.up(low));x=low
        parts.append(x)
        return parts[::-1] if self.order=='coarse_first' else parts
    def decode(self,parts):
        if isinstance(parts,dict):parts=[parts[k] for k in sorted(parts)]
        p=parts if self.order=='fine_first' else parts[::-1]
        x=p[-1]
        for r in p[-2::-1]:x=self.op.up(x)+r
        return x

@register('representation','pyramid')
class Pyramid:
    def __init__(self,factors=(1,2),mode='average',order='fine_first'):
        self.factors=tuple(factors);self.mode=mode;self.order=order
        if self.factors[0]!=1:raise ValueError('Pyramid first factor must be one')
    def encode(self,x):
        out=[x];prev=1
        for f in self.factors[1:]:
            if f%prev:raise ValueError('Nonintegral pyramid ratios')
            x=block_average(x,f//prev) if self.mode=='average' else resize(x,scale=prev/f,mode=self.mode)
            out.append(x);prev=f
        return out if self.order=='fine_first' else out[::-1]
    def decode(self,parts):return parts[0] if self.order=='fine_first' else parts[-1]

@register('representation','channel_permutation')
class ChannelPermutation:
    def __init__(self,factor=2,width_only=False):self.factor=factor;self.width_only=width_only
    def encode(self,x):return channel_down(x,self.factor,self.width_only)
    def decode(self,x):return channel_up(x,self.factor,self.width_only)

@register('representation','module_codec')
class ModuleCodec:
    """External VAE/SR without any implicit download. Modules must be supplied."""
    def __init__(self,encoder,decoder,scale=1.,shift=0.):
        self.encoder=encoder;self.decoder=decoder;self.scale=scale;self.shift=shift
    def encode(self,x):return (self.encoder(x)-self.shift)*self.scale
    def decode(self,z):return self.decoder(z/self.scale+self.shift)
