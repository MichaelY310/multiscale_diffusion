"""Inference policies for an already trained data-to-noise flow backbone.

They own no training objective and always address model stage 0. Applying them
to a stage-trained model with different clocks/heads requires another adapter.
"""
import math
import torch
from .registry import register
from .operators import resize
from .trajectories import Trace


def lanczos_resize(x, target_hw, radius=3):
    """Separable, half-pixel Lanczos with downsampling antialias and renormalized
    truncated boundaries. Floating tensor policy is explicit (no PIL clipping).
    """
    if radius<1:raise ValueError('Lanczos radius must be positive')
    for axis,target in zip((-2,-1),target_hw):
        source=x.shape[axis]
        if source==target:continue
        scale=min(target/source,1.)
        coord=(torch.arange(target,device=x.device,dtype=torch.float64)+.5)*source/target-.5
        distance=(coord[:,None]-torch.arange(source,device=x.device,dtype=torch.float64)[None])*scale
        weights=torch.sinc(distance)*torch.sinc(distance/radius)*(distance.abs()<radius)
        weights=(weights/weights.sum(-1,keepdim=True)).to(x)
        x=torch.matmul(x.movedim(axis,-1),weights.T).movedim(-1,axis)
    return x


@register('sampler','bottleneck_grid')
class BottleneckGrid:
    """Paper Eq.7-9 with explicit integer start indices.

    start_indices implements the prose/Eq.7 retained N*w steps convention.
    Algorithm 1 line 12 prints the opposite (1-w)*N; we do not hide this
    ambiguity behind a claimed author-default strength parameter.
    """
    def __init__(self,resolutions,steps,start_indices,shifts,interpolation='lanczos',lanczos_radius=3):
        n=len(resolutions)
        if n<1 or len(steps)!=n or len(start_indices)!=n or len(shifts)!=n:raise ValueError('One resolution/count/start/shift per stage is required')
        self.resolutions=[tuple(x) for x in resolutions]
        if any(len(x)!=2 or min(x)<1 for x in self.resolutions):raise ValueError('Bottleneck keeps T and changes H,W only')
        if any(not 0<=a<s for a,s in zip(start_indices,steps)) or min(shifts)<=0 or start_indices[0]!=0:raise ValueError('Invalid stage restart schedule')
        if interpolation not in ('lanczos','bilinear','bicubic','nearest'):raise ValueError('Unknown resampling kernel')
        self.steps,self.starts,self.shifts=steps,start_indices,shifts
        self.interpolation,self.radius=interpolation,lanczos_radius

    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        if tuple(initial.shape[-2:])!=self.resolutions[0]:raise ValueError('Initial noise grid must equal the first Bottleneck grid')
        state=initial;trace=Trace(return_trajectory,state);kwargs=model_kwargs or {}
        for i,(size,count,start,shift) in enumerate(zip(self.resolutions,self.steps,self.starts,self.shifts)):
            raw=torch.linspace(1,0,count+1,device=state.device,dtype=state.dtype)
            schedule=(shift*raw/(1+(shift-1)*raw))[start:]
            if i:
                state=lanczos_resize(state,size,self.radius) if self.interpolation=='lanczos' else resize(state,size=size,mode=self.interpolation)
                state=(1-schedule[0])*state+schedule[0]*stream.normal(state)
                trace.append(state)
            for t,s in zip(schedule[:-1],schedule[1:]):
                state=state+(s-t)*model(state,t.expand(len(state)),stage=0,**kwargs)
                trace.append(state)
        return (state,list(trace)) if return_trajectory else state


@register('sampler','speed_grid')
class SPEEDGrid:
    """Existing validated spectral boundary on any grid flow backbone.

    Boundaries/steps are explicit, not a fitted spectrum or a quality-certified
    FLUX/Wan scheduler. T is preserved; frames share the joint denoising model.
    """
    def __init__(self,resolutions,steps,boundaries,transform='dct'):
        from .boundaries import SpectralBoundary
        n=len(resolutions)
        if len(steps)!=n or min(steps)<1 or len(boundaries)!=n+1 or boundaries[0]!=1 or boundaries[-1]!=0 or any(a<=b for a,b in zip(boundaries,boundaries[1:])):raise ValueError('Invalid SPEED stage schedule')
        self.resolutions=[tuple(x) for x in resolutions];self.steps=steps;self.bounds=boundaries
        if any(len(x)!=2 or min(x)<1 for x in self.resolutions):raise ValueError('SPEED uses spatial grids')
        self.boundary=SpectralBoundary(transform)
        for a,b in zip(self.resolutions[:-1],self.resolutions[1:]):
            if b[0]<a[0] or b[0]/a[0]!=b[1]/a[1]:raise ValueError('SPEED requires isotropic expanding spatial grids')
    def initial_shape(self,shape):return [*shape[:-2],*self.resolutions[0]]

    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        if tuple(initial.shape[-2:])!=self.resolutions[0]:raise ValueError('Incorrect SPEED initial grid')
        state=initial;trace=Trace(return_trajectory,state);kwargs=model_kwargs or {}
        for i,count in enumerate(self.steps):
            start=self.bounds[i]
            if i:
                ratio=self.resolutions[i][0]/self.resolutions[i-1][0]
                seed=int(stream.randint(0,2**31-1,()).item())
                result=self.boundary(state,time=start,seed=seed,target_hw=self.resolutions[i],ratio=ratio)
                state,start=result.state,result.time;trace.append(state)
            grid=torch.linspace(start,self.bounds[i+1],count+1,device=state.device,dtype=state.dtype)
            for t,s in zip(grid[:-1],grid[1:]):
                state=state+(s-t)*model(state,t.expand(len(state)),stage=0,**kwargs)
                trace.append(state)
        return (state,list(trace)) if return_trajectory else state
