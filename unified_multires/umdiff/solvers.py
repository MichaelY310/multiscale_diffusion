"""Explicit solver arithmetic; saved trajectories expose every update."""
import numpy as np
import torch
from .tree import map_tree,first
from .registry import register

@register('solver','euler')
class Euler:
    def __init__(self,accumulation='model',step_dtype='model'):
        if accumulation not in ('model','float32','float64'):raise ValueError('Invalid accumulator dtype')
        self.accumulation=accumulation;self.step_dtype=step_dtype
    def step(self,state,velocity,dt):
        def update(x,v):
            dtype=x.dtype if self.accumulation=='model' else getattr(torch,self.accumulation)
            d=dt.to(dtype=v.dtype if self.step_dtype=='model' else getattr(torch,self.step_dtype),device=x.device) if torch.is_tensor(dt) else dt
            return (x.to(dtype)+d*v).to(v.dtype if self.accumulation!='float64' else torch.float64)
        return map_tree(update,state,velocity)

@register('solver','heun')
class Heun(Euler):
    def integrate(self,model,state,t0,t1):
        v0=model(state,t0);trial=self.step(state,v0,t1-t0);v1=model(trial,t1)
        return self.step(state,map_tree(lambda a,b:(a+b)*.5,v0,v1),t1-t0)

@register('solver','ddpm')
class DDPM:
    """ADM fixed-small posterior, including released RDD drift coefficient."""
    def __init__(self,clip_clean=True):self.clip_clean=clip_clean
    def posterior(self,path,clean,state,t):
        if self.clip_clean:clean=clean.clamp(-1,1)
        b=path.betas;a=1-b;abar=path.abar;prev=np.append(1.,abar[:-1])
        c1=b*np.sqrt(prev)/(1-abar);c2=(1-prev)*np.sqrt(a)/(1-abar)
        variance=b*(1-prev)/(1-abar)
        mean=path.extract(c1,t,state)*clean+path.extract(c2,t,state)*state
        if path.drift_start is not None:
            mapped=path.timestep_map
            cd=np.sqrt(prev)/(1-abar)*((mapped-path.drift_start)-a*(mapped-path.drift_start)-a+abar)/(path.drift_end-path.drift_start)
            mean=mean+path.multiplier*path.extract(cd,t,state)*(path.projection(clean)-clean)
        # Source uses clipped log variance, not sqrt(var), at t=0 too.
        logvar=np.log(np.append(variance[1],variance[1:])) if len(b)>1 else np.array([np.log(max(variance[0],1e-20))])
        return mean,path.extract(logvar,t,state).expand_as(state)
    def step(self,path,clean,state,t,noise):
        mean,logvar=self.posterior(path,clean,state,t)
        active=(t!=0).to(state.dtype).reshape(-1,*([1]*(state.ndim-1)))
        return mean+active*torch.exp(.5*logvar)*noise
