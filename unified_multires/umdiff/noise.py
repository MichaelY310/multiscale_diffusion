"""Noise covariance and cross-resolution coupling, separate from path means."""
import torch
from .registry import register
from .operators import Projection,repeat,rolling_noise
from .tree import map_tree

@register('noise','iid')
class IID:
    def __call__(self,clean,stream):return map_tree(stream.normal,clean)

@register('noise','relay')
class RelayNoise:
    def __init__(self,block=4,block_scale=0.):self.block=block;self.scale=block_scale
    def __call__(self,clean,stream):
        # Native BlurLoss consumes a discarded initial draw if block_scale>0.
        noise=stream.normal(clean)
        if self.scale>0:
            noise=stream.normal(clean)+self.scale*rolling_noise(stream.normal(clean),self.block)
        return noise

@register('noise','decomposed')
class Decomposed:
    def __init__(self,representation):
        from .registry import build
        self.representation=build('representation',representation)
    def __call__(self,clean,stream):
        image=self.representation.decode(clean)
        return self.representation.encode(stream.normal(image))

@register('noise','lapflow_native')
class LapFlowNative:
    def __init__(self,representation):
        from .registry import build
        self.representation=build('representation',representation)
    def __call__(self,clean,stream):
        if len(clean)==2:
            return self.representation.encode(stream.normal(self.representation.decode(clean)))
        if len(clean)==3:return [stream.normal(v) for v in clean]
        raise ValueError('Released LapFlow implements only two or three levels')

def coupled_lift(coarse,low_innovation,high_innovation,factor=2,rho=1.):
    """Q=U/factor is an isometry in 2D; output marginal covariance I.

    Cov(epsilon_fine,epsilon_coarse)=rho Q. All input innovations IID.
    """
    if not -1<=rho<=1:raise ValueError('Invalid correlation')
    p=Projection(factor=factor)
    q=lambda x:repeat(x,factor)/factor
    return rho*q(coarse)+(1-rho*rho)**.5*q(low_innovation)+p.high(high_innovation)

def equicorrelated_blocks(x,gamma=-1/3,jitter=1e-6,standard=None,stream=None):
    """Native PixelFlow 2x2 block covariance, with explicit standard normals.

    CPU native loop order is B,C,H-block,W-block, each draw shape (4,).
    """
    if x.shape[-2]%2 or x.shape[-1]%2:raise ValueError('Even spatial dimensions required')
    cov=torch.eye(4,device=x.device,dtype=x.dtype)*(1-gamma+jitter)+torch.ones(4,4,device=x.device,dtype=x.dtype)*gamma
    chol=torch.linalg.cholesky(cov)
    out=torch.empty_like(x)
    for b in range(x.shape[0]):
        for c in range(x.shape[1]):
            for i in range(x.shape[-2]//2):
                for j in range(x.shape[-1]//2):
                    z=(torch.randn(4,device=x.device,dtype=x.dtype) if stream is None else stream.normal(torch.empty(4,device=x.device,dtype=x.dtype))) if standard is None else standard[b,c,i,j]
                    out[b,c,2*i:2*i+2,2*j:2*j+2]=(chol@z).reshape(2,2)
    return out
