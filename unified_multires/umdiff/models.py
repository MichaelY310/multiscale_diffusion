"""A usable small backbone, plus explicit source-native adapter hooks.

TinyConv is for component experiments, never labeled as an author's backbone.
"""
import math
import torch
from .registry import register,build

@register('model','ensemble')
class Ensemble(torch.nn.ModuleDict):
    """Named native models with explicit ownership and checkpoint keys."""
    def __init__(self,models):super().__init__({key:build('model',spec) for key,spec in models.items()})

@register('model','tiny_conv')
class TinyConv(torch.nn.Module):
    def __init__(self,channels=3,width=32,heads=None):
        super().__init__();self.heads=heads
        outputs=channels*(len(heads) if heads else 1)
        self.time=torch.nn.Sequential(torch.nn.Linear(16,width),torch.nn.SiLU(),torch.nn.Linear(width,width))
        self.input=torch.nn.Conv2d(channels,width,3,padding=1)
        self.body=torch.nn.Sequential(torch.nn.GroupNorm(4,width),torch.nn.SiLU(),torch.nn.Conv2d(width,width,3,padding=1),torch.nn.SiLU())
        self.output=torch.nn.Conv2d(width,outputs,3,padding=1)
    def forward(self,x,t,**kwargs):
        if t.ndim==0:t=t.expand(x.shape[0])
        frequency=torch.exp(-math.log(10000)*torch.arange(8,device=x.device,dtype=torch.float32)/7)
        phase=t.float().reshape(-1,1)*frequency.reshape(1,-1)
        emb=torch.cat([torch.sin(phase),torch.cos(phase)],dim=-1).to(x.dtype)
        h=self.input(x)+self.time(emb)[:,:,None,None]
        out=self.output(h+self.body(h))
        if self.heads:return dict(zip(self.heads,out.chunk(len(self.heads),dim=1)))
        return out

@register('model','product_conv')
class ProductConv(torch.nn.Module):
    """Separate networks per product-state leaf; a controlled architecture."""
    def __init__(self,channels=(3,3),width=32,heads=None):
        super().__init__();self.models=torch.nn.ModuleList([TinyConv(c,width,heads) for c in channels])
    def forward(self,x,t,**kwargs):
        if len(x)!=len(self.models):raise ValueError('Product leaf count differs from model')
        out=[]
        for i,(value,model) in enumerate(zip(x,self.models)):
            local=t[:,i] if t.ndim==2 else t
            out.append(model(value,local,**kwargs))
        return out

@register('model','python_factory')
def python_factory(module,factory,args=None):
    import importlib
    return getattr(importlib.import_module(module),factory)(**(args or {}))
