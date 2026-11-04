"""Relay's heat-coordinate stochastic predictor/corrector sampler."""
import math
import numpy as np
import scipy.stats as stats
import torch
from .operators import block_dct,rolling_noise
from .registry import register

@register('sampler','relay')
class RelaySampler:
    def __init__(self,steps=20,sigma_min=.008,sigma_max=80.,rho=7.,truncation_sigma=.9,truncation_t=.93,patch=4,block_scale=.15,stochasticity=.2,blur_max=3.,cfg=1.,dct_backend='fft',p_mean=-1.2,p_std=1.2,min_scale=.001):
        self.steps=steps;self.minimum=sigma_min;self.maximum=sigma_max;self.rho=rho;self.truncation=truncation_sigma;self.prob=truncation_t;self.patch=patch;self.block=block_scale;self.stochasticity=stochasticity;self.blur=blur_max;self.cfg=cfg
        self.dct_backend=dct_backend
        self.mean=p_mean;self.std=p_std;self.floor=min_scale
        if steps<2 or not 0<=stochasticity<=1:raise ValueError('Invalid Relay sampler configuration')
    def alpha(self,sigma,state):
        t=stats.norm.cdf((np.log(sigma.cpu())-self.mean)/self.std)/self.prob
        t=torch.tensor(t,device=state.device)
        blur=self.blur*torch.sin(t*torch.pi/2)**2
        freq=torch.pi*torch.linspace(0,self.patch-1,self.patch,device=state.device)/self.patch
        freq2=freq[:,None]**2+freq[None,:]**2
        return torch.exp(-freq2.repeat(state.shape[-2]//self.patch,state.shape[-1]//self.patch)*(blur**2/2))*(1-self.floor)+self.floor
    @torch.no_grad()
    def __call__(self,net,latents,labels=None,randn_like=torch.randn_like,return_trajectory=False,*,stream=None,model_kwargs=None):
        if stream is not None:randn_like=stream.normal
        if model_kwargs:
            kwargs=dict(model_kwargs);labels=kwargs.pop('class_labels',kwargs.pop('labels',labels))
            if kwargs:raise ValueError('Relay sampler accepts labels/class_labels only')
        minimum=max(self.minimum,net.sigma_min);maximum=min(self.maximum,net.sigma_max)
        indices=torch.arange(self.steps,dtype=torch.float64,device=latents.device)
        schedule=(maximum**(1/self.rho)+indices/(self.steps-1)*(minimum**(1/self.rho)-maximum**(1/self.rho)))**self.rho
        below=torch.where(schedule<self.truncation)[0]
        if not len(below):raise ValueError('Truncation removes every Relay step')
        schedule=schedule[int(below[0]):];schedule=torch.cat([net.round_sigma(schedule),torch.zeros_like(schedule[:1])])
        state=latents.double();trajectory=[]
        def denoise(x,t):
            conditional=net(x,t,labels).double()
            if self.cfg>1:return self.cfg*conditional-(self.cfg-1)*net(x,t,None).double()
            return conditional
        def noise(ref):
            value=randn_like(ref)
            if self.block>0:value=value+self.block*rolling_noise(randn_like(latents),self.patch)
            return value
        for i,(sigma,next_sigma) in enumerate(zip(schedule[:-1],schedule[1:])):
            if i==0:
                # Native initial addition scales IID and block terms separately.
                state=state+randn_like(state)*sigma
                if self.block>0:state=state+self.block*rolling_noise(randn_like(latents),self.patch)*sigma
            current=state;clean=denoise(current,sigma)
            if i==len(schedule)-2:
                state=clean
                if return_trajectory:trajectory.append(state.clone())
                break
            ac=self.alpha(sigma,state);an=self.alpha(next_sigma,state)
            transform=lambda value,inverse=False:block_dct(value,self.patch,inverse,self.dct_backend)
            u=transform(current);u0=transform(clean);d=(u-u0)/sigma
            gamma=math.sqrt(1-self.stochasticity**2)*next_sigma/sigma
            factor=an+gamma-gamma*ac;drift=sigma*(gamma*ac-an)
            state=transform(factor*u+drift*d,True)+self.stochasticity*next_sigma*noise(state)
            predicted=denoise(state,next_sigma)
            dnext=(transform(state)-transform(predicted))/next_sigma
            # Native corrector consumes fresh stochastic draws again.
            state=transform(factor*u+drift*(d+dnext)/2,True)+self.stochasticity*next_sigma*noise(state)
            if return_trajectory:trajectory.append(state.clone())
        return (state,trajectory) if return_trajectory else state

@register('sampler','relay_chain')
class RelayChainSampler:
    """Two native networks: low-resolution EDM -> resize -> blur diffusion."""
    def __init__(self,output_size=256,first=None,second=None):
        self.size=output_size;self.first=first or {};self.second=RelaySampler(**(second or {}))
    @torch.no_grad()
    def __call__(self,models,initial,stream,model_kwargs=None,return_trajectory=False):
        from . import native
        from .operators import resize
        if set(models.keys())!={'low','high'}:raise ValueError('Relay chain requires named low/high models')
        kwargs=dict(model_kwargs or {});labels=kwargs.pop('class_labels',kwargs.pop('labels',None))
        if kwargs:raise ValueError('Unknown Relay conditioning')
        edm=native.extract('relay/generate_imagenet.py',['edm_sampler'],dict(torch=torch,np=np))['edm_sampler']
        low=edm(models['low'],initial,class_labels=labels,randn_like=stream.normal,**self.first)
        # Native F.interpolate(images,256) defaults to nearest.
        high=resize(low,size=(self.size,self.size),mode='nearest') if low.ndim==5 else torch.nn.functional.interpolate(low,self.size)
        output,trace=self.second(models['high'],high,labels=labels,stream=stream,return_trajectory=True)
        return (output,[low,high,*trace]) if return_trajectory else output

@register('sampler','edm')
class EDMSampler:
    def __init__(self,**kwargs):self.kwargs=kwargs
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        if return_trajectory:raise ValueError('Native EDM backend exposes final state only')
        from . import native
        function=native.extract('relay/generate_imagenet.py',['edm_sampler'],dict(torch=torch,np=np))['edm_sampler']
        kwargs=dict(model_kwargs or {});labels=kwargs.pop('class_labels',None)
        if kwargs:raise ValueError('EDM accepts class_labels only')
        return function(model,initial,class_labels=labels,randn_like=stream.normal,**self.kwargs)
