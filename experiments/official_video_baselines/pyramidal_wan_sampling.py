"""Published global-sigma Euler path; optional stage-local solver ablations.

The exact official v0.33.0 scheduler is vendored without modification. Separate
first-frame/tail sigma grids share one natural conditioning level. The paper
specifies spectral boundaries and scheduler, but not the dense grid within each
stage; uniform natural grids (shift=1) are an explicitly disclosed choice.
"""
from functools import lru_cache
import importlib.util
from pathlib import Path
import sys
import torch
from pyramidal_wan_paper_math import (NATURAL_CLEAN,NATURAL_NOISY,noise_at_stage,
                                     sigma_tensor,to_global,transition,up)


@lru_cache(maxsize=1)
def scheduler_class():
    path=Path(__file__).resolve().parents[3]/'paper_sources_20261006/diffusers_v0330/scheduling_flow_match_euler_discrete.py'
    name='diffusers.schedulers._rdd_official_v0330_flow_euler'
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.FlowMatchEulerDiscreteScheduler


def stage_schedulers(stage,count,device):
    if count<1:raise ValueError('Positive updates required')
    natural=torch.linspace(NATURAL_NOISY[stage],NATURAL_CLEAN[stage],count+1,device=device,dtype=torch.float32)
    schedulers=[]
    for first in (True,False):
        sigma=to_global(natural,stage,first)
        scheduler=scheduler_class()(shift=1.)
        # Include the cleaner endpoint as the next sigma for the final actual
        # update. v0.33.0 always appends zero; that extra step is never executed.
        scheduler.set_timesteps(sigmas=sigma.cpu().tolist(),device=device)
        schedulers.append(scheduler)
    return natural,schedulers


@torch.no_grad()
def sample(transformer,full_shape,text,negative,*,counts=(20,20,10),seed=42,cfg=5.,solver='euler'):
    if solver not in ('euler','ab2','heun') or len(counts)!=3:
        raise ValueError('Three coarse/middle/fine update counts and a supported solver required')
    device=text.device;generator=torch.Generator(device=device).manual_seed(seed)
    full_noise=torch.randn(full_shape,device=device,generator=generator,dtype=torch.float32)
    x=noise_at_stage(full_noise,2)
    evaluations=calls=0;grids=[]

    def velocity(state,natural):
        nonlocal evaluations,calls
        z=natural.reshape(1)
        with torch.autocast(device.type,dtype=torch.bfloat16):
            conditional=transformer([state],[z],[text])[0].float()
            calls+=1
            if cfg!=1:
                unconditional=transformer([state],[z],[negative])[0].float();calls+=1
                conditional=unconditional+cfg*(conditional-unconditional)
        evaluations+=1
        return conditional

    for stage,count in zip((2,1,0),counts):
        natural,schedulers=stage_schedulers(stage,count,device)
        grids.append({'paper_stage':stage,'natural':natural.cpu().tolist(),
            'first_global_sigma':schedulers[0].sigmas[:count+1].cpu().tolist(),
            'tail_global_sigma':schedulers[1].sigmas[:count+1].cpu().tolist()})
        previous=previous_delta=None
        for n in range(count):
            flow=velocity(x,natural[n])
            delta=sigma_tensor(x,natural[n+1],stage)-sigma_tensor(x,natural[n],stage)
            if solver=='euler':
                first=schedulers[0].step(flow[:,:,:1],schedulers[0].timesteps[n],x[:,:,:1],return_dict=False)[0]
                tail=schedulers[1].step(flow[:,:,1:],schedulers[1].timesteps[n],x[:,:,1:],return_dict=False)[0]
                x=torch.cat([first,tail],dim=2)
            elif solver=='heun':
                next_flow=velocity(x+delta*flow,natural[n+1]);x=x+delta*(flow+next_flow)/2
            else:
                ratio=0. if previous is None else delta/(2*previous_delta)
                effective=flow if previous is None else (1+ratio)*flow-ratio*previous
                x=x+delta*effective;previous,previous_delta=flow,delta
            if not torch.isfinite(x).all():raise FloatingPointError('Nonfinite paper Wan trajectory')
        if stage>0:
            lifted=up(x)
            innovation=torch.randn(lifted.shape,device=device,generator=generator,dtype=torch.float32)
            x=transition(x,stage,innovation)
    if tuple(x.shape)!=tuple(full_shape):raise AssertionError('Paper output shape changed')
    return x,{'solver':solver,'solver_steps_coarse_mid_fine':list(counts),
        'scheduled_updates':sum(counts),'velocity_evaluations':evaluations,
        'branch_forward_calls':calls,'cfg':cfg,'boundary_velocity_evaluations':0,
        'history_reset_at_every_boundary':True,'stage_grids':grids,
        'scheduler':'official diffusers v0.33.0 FlowMatchEulerDiscreteScheduler, shift=1',
        'dense_grid_choice':'uniform natural level within each published stage; author grid undisclosed'}
