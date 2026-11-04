"""Independent SPEED PixelGen and native-Wan-interface sampling loops.

Model implementations and weights are supplied by the caller. These adapters
never load assets. CPU runs do not certify the author's CUDA autocast kernels.
"""
import types
import numpy as np
import torch
import torch.nn.functional as F
from ..registry import register
from ..boundaries import SpectralBoundary


def stage_indices(sigmas, thresholds, steps):
    return [next((i for i, v in enumerate(sigmas) if v <= t), steps) for t in thresholds]


def validate_schedule(scales, transitions):
    if not scales or scales[-1] != 1 or any(a >= b for a, b in zip(scales, scales[1:])) or scales[0] <= 0:
        raise ValueError('SPEED scales must increase and end at one')
    if len(transitions) != len(scales)-1:
        raise ValueError('One transition per scale boundary is required')


def pixelgen_position(model, resolution, trained_grid):
    grid = resolution // model.patch_size
    if grid ** 2 != model.pos_embed.shape[1]:
        pos = np.linspace(0., float(trained_grid-1), grid, dtype=np.float32)
        half = model.hidden_size // 2
        omega = 1. / (10000. ** (np.arange(half//2, dtype=np.float64)/(half/2.)))
        h, w = np.meshgrid(pos, pos)
        a = np.einsum('m,d->md', h.reshape(-1), omega)
        b = np.einsum('m,d->md', w.reshape(-1), omega)
        pe = np.concatenate([np.sin(a), np.cos(a), np.sin(b), np.cos(b)], axis=1).astype(np.float32)
        model.pos_embed = torch.nn.Parameter(torch.from_numpy(pe).unsqueeze(0).to(model.pos_embed), requires_grad=False)
        model.num_patches = grid ** 2
    model.decoder_patch_scaling_h = resolution / 512.
    model.decoder_patch_scaling_w = resolution / 512.


def attach_pixelgen_rope(model, trained_grid, frequency_function):
    """Supply PixelGen's frequency primitive explicitly, retaining its cache API."""
    model.precompute_pos = {}
    def fetch_pos(self, height, width, device):
        key = (height, width)
        if key not in self.precompute_pos:
            self.precompute_pos[key] = frequency_function(
                self.hidden_size // self.num_groups, height, width, theta=10000.,
                scale=(trained_grid/float(height), trained_grid/float(width))).to(device)
        return self.precompute_pos[key].to(device)
    model.fetch_pos = types.MethodType(fetch_pos, model)


@register('sampler', 'speed_pixelgen')
class SPEEDPixelGenSampler:
    def __init__(self, full_hw=(512,512), scales=(.5,1.), transitions=(.5,), steps=12,
                 transform='dct', seed=0, guidance=4., timeshift=1., trained_grid=32):
        validate_schedule(scales, transitions)
        if full_hw[0] != full_hw[1]:
            raise ValueError('Native PixelGen is square-only')
        self.hw=full_hw; self.scales=scales; self.transitions=transitions; self.steps=steps
        self.transform=transform; self.seed=seed; self.guidance=guidance
        self.shift=timeshift; self.grid=trained_grid

    @torch.no_grad()
    def __call__(self, model, initial, stream=None, model_kwargs=None, return_trajectory=False):
        kwargs=dict(model_kwargs or {}); cond=kwargs.pop('cond'); uncond=kwargs.pop('uncond')
        if kwargs: raise ValueError(f'Unknown PixelGen conditioning: {set(kwargs)}')
        ts=torch.linspace(0.,1.-1./self.steps,self.steps)
        ts=torch.cat([ts,torch.tensor([1.])]); ts=ts/(ts+(1.-ts)*self.shift)
        dts=(ts[1:]-ts[:-1]).to(initial.device); ts=ts.to(initial.device)
        sigmas=1.-ts; indices=stage_indices(sigmas.tolist(),self.transitions,self.steps)
        state=initial; t_cur=torch.zeros([1],device=initial.device); trace=[state]
        pixelgen_position(model,round(self.scales[0]*self.hw[0]),self.grid)
        for stage,(start,end,scale) in enumerate(zip([0]+indices,indices+[self.steps],self.scales)):
            for i in range(start,end):
                x=torch.cat([state,state]); t=t_cur.repeat(2); text=torch.cat([uncond,cond])
                with torch.autocast(device_type=initial.device.type,dtype=torch.bfloat16,enabled=initial.device.type=='cuda'):
                    clean=model(x,t,text)
                v=(clean-x)/(1.-t.view(-1,1,1,1)).clamp_min(5e-2)
                vu,vc=v.chunk(2); velocity=vu+self.guidance*(vc-vu)
                state=state+velocity*dts[i]; t_cur=t_cur+dts[i]; trace.append(state)
            if stage+1<len(self.scales):
                next_scale=self.scales[stage+1]; side=round(next_scale*self.hw[0])
                boundary=SpectralBoundary(self.transform)(state,target_hw=(side,side),
                    ratio=next_scale/scale,time=sigmas[end].item() if end<self.steps else 0.,
                    seed=self.seed+(stage+1)*10000)
                state=boundary.state; ts[end]=1.-boundary.time
                if end<self.steps:
                    dts[end-1]=ts[end]-ts[end-1]; dts[end]=ts[end+1]-ts[end]
                t_cur=torch.tensor([1.-boundary.time],device=state.device,dtype=t_cur.dtype)
                pixelgen_position(model,side,self.grid); trace.append(state)
        return (state,trace) if return_trajectory else state


@register('sampler','speed_wan')
class SPEEDWanSampler:
    def __init__(self, full_hw=(60,104), scales=(.5,1.), transitions=(.5,), steps=12,
                 transform='dct', seed=0, guidance=5., shift=3.):
        validate_schedule(scales,transitions)
        self.hw=full_hw; self.scales=scales; self.transitions=transitions; self.steps=steps
        self.transform=transform; self.seed=seed; self.guidance=guidance; self.shift=shift

    @torch.no_grad()
    def __call__(self,model,initial,stream=None,model_kwargs=None,return_trajectory=False):
        from diffusers import FlowMatchEulerDiscreteScheduler
        kwargs=dict(model_kwargs or {}); cond=kwargs.pop('ctx_cond'); uncond=kwargs.pop('ctx_uncond')
        if kwargs: raise ValueError(f'Unknown Wan conditioning: {set(kwargs)}')
        raw=np.linspace(1.,0.,self.steps+1)[:-1]
        shifted=self.shift*raw/(1.+(self.shift-1.)*raw)
        scheduler=FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        scheduler.set_timesteps(device=initial.device,sigmas=shifted.tolist())
        sigmas=scheduler.sigmas; indices=stage_indices(sigmas,self.transitions,self.steps)
        state=initial; trace=[state]
        for stage,(start,end,scale) in enumerate(zip([0]+indices,indices+[self.steps],self.scales)):
            for i in range(start,end):
                h,w=state.shape[-2:]; ph,pw=h%2,w%2
                x=F.pad(state,(0,pw,0,ph)) if ph or pw else state
                seq=int(np.ceil(x.shape[-2]*x.shape[-1]/4*x.shape[-3]))
                t=torch.stack([scheduler.timesteps[i]])
                with torch.autocast(device_type=state.device.type,dtype=torch.bfloat16,enabled=state.device.type=='cuda'):
                    vc=model([x],t=t,context=cond,seq_len=seq)[0]
                    vu=model([x],t=t,context=uncond,seq_len=seq)[0]
                v=vu+self.guidance*(vc-vu)
                if ph or pw:v=v[...,:h,:w]
                state=scheduler.step(v.unsqueeze(0),scheduler.timesteps[i],state.unsqueeze(0),return_dict=False)[0].squeeze(0)
                trace.append(state)
            if stage+1<len(self.scales):
                nxt=self.scales[stage+1]
                boundary=SpectralBoundary(self.transform)(state,target_hw=tuple(round(nxt*n) for n in self.hw),
                    ratio=nxt/scale,time=sigmas[end].item() if end<self.steps else 0.,seed=self.seed+(stage+1)*10000)
                scheduler.sigmas[end]=boundary.time; scheduler.timesteps[end]=boundary.time*1000.
                scheduler._step_index=None; scheduler._begin_index=None; scheduler.set_begin_index(end)
                state=boundary.state.to(dtype=torch.bfloat16); trace.append(state)
        return (state,trace) if return_trajectory else state
