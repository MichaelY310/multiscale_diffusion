"""Native RDD solvers as a backend; composable stage boundaries remain explicit."""
import importlib,importlib.util,sys
import numpy as np
import torch
from . import native
from .locations import rdd_root
from .registry import register
from .boundaries import RDDFresh,Isometric

def diffusion_backend():
    root=rdd_root()
    existing=sys.modules.get('diffusion')
    if existing is not None and not str(getattr(existing,'__file__','')).startswith(str(root)):
        raise RuntimeError('RDD native namespace collision; use a separate process')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return importlib.import_module('diffusion')

@register('model','native_rdd_unified')
def rdd_unified(**kwargs):
    path=rdd_root()/'DiT/models_multires.py'
    spec=importlib.util.spec_from_file_location('umdiff_rdd_multires',path);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module.DiTMultiRes(**kwargs)

@register('sampler','rdd_native_stages')
class NativeRDDStages:
    def __init__(self,resolutions=(32,16,8),stage_lengths=(300,300,400),steps=(100,75,75),solver='dpm2',learned_variance=True,boundary='fresh',guidance_scale=1.,null_label=1000,noise_schedule='linear',clip_clean=False,drift_scale=1.,sigma_h=0.,guidance_scale_per_stage=None):
        if len(resolutions)!=len(stage_lengths) or len(steps)!=len(resolutions):raise ValueError('One length/step count per stage required')
        if solver not in ('ddpm','jump','ancestral','ddim','dpm1','dpm2'):raise ValueError('Unknown RDD solver')
        boundary={'refresh':'fresh','coupled':'noise_filling','isometric':'legacy_initial_noise'}.get(boundary,boundary)
        if boundary not in ('fresh','noise_filling','legacy_initial_noise','bilinear'):raise ValueError('RDD boundary must be fresh, noise_filling or legacy_initial_noise')
        if boundary=='noise_filling' and solver in ('ddim','ancestral'):raise ValueError('Recovered-noise filling requires a jump/DDPM or DPM endpoint; DDIM/ancestral terminal clocks differ')
        self.resolutions=list(resolutions);self.steps=steps;self.solver=solver;self.boundary=boundary;self.cfg=guidance_scale;self.null=null_label;self.clip=clip_clean
        self.drift_scale=drift_scale;self.sigma_h=sigma_h;self.stage_cfg=guidance_scale_per_stage
        if self.stage_cfg is not None and (len(self.stage_cfg)!=len(resolutions) or self.cfg<=1):raise ValueError('Per-stage CFG needs one scale per stage and guidance_scale>1 to activate the doubled-batch path')
        module=diffusion_backend();limits=np.concatenate([[0],np.cumsum(stage_lengths)]);self.diffusions=[]
        for i,resolution in enumerate(resolutions):
            self.diffusions.append(module.create_diffusion_with_drift(str(steps[len(steps)-1-i]),noise_schedule=noise_schedule,diffusion_steps=int(limits[-1]),learn_sigma=learned_variance,c=i if i<len(resolutions)-1 else None,T_c=int(limits[i]),T_c_plus_1=int(limits[i+1]),sample_scale=resolution//resolutions[i+1] if i<len(resolutions)-1 else 2))
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        if initial.shape[-2:]!=(self.resolutions[-1],)*2:raise ValueError('RDD initial state must use the coarsest grid')
        kwargs=dict(model_kwargs or {});net=model.model if hasattr(model,'model') else model;n=len(initial);trace=[];state=initial
        forward=net.forward
        if self.cfg>1:
            if 'y' not in kwargs:raise ValueError('RDD guidance requires labels')
            kwargs['y']=torch.cat([kwargs['y'],torch.full_like(kwargs['y'],self.null)]);kwargs['cfg_scale']=self.cfg;forward=net.forward_with_cfg
        # Backend solvers call the global Torch RNG. Its complete state is scoped
        # to the supplied stream; no patches to native functions are installed.
        with stream.native_rng(initial.device):
            if self.cfg>1:state=torch.cat([state,torch.randn_like(state)])
            coarse_noise=state[:n]
            for position,index in enumerate(range(len(self.resolutions)-1,-1,-1)):
                diffusion=self.diffusions[index]
                diffusion.drift_scale_multiplier=self.drift_scale
                stage_net=net.for_resolution(self.resolutions[index]) if hasattr(net,'for_resolution') else net
                forward=stage_net.forward_with_cfg if self.cfg>1 else stage_net.forward
                if self.stage_cfg is not None:kwargs['cfg_scale']=self.stage_cfg[position]
                if position:
                    ratio=self.resolutions[index]//self.resolutions[index+1];t=torch.full((n,),diffusion.num_timesteps-1,device=initial.device,dtype=torch.long)
                    alpha=torch.from_numpy(diffusion.sqrt_alphas_cumprod).to(initial.device)[t].float().reshape(n,1,1,1)
                    sigma=torch.from_numpy(diffusion.sqrt_one_minus_alphas_cumprod).to(initial.device)[t].float().reshape(n,1,1,1)
                    fine=torch.empty(n,initial.shape[1],self.resolutions[index],self.resolutions[index],device=initial.device,dtype=clean.dtype)
                    if self.boundary=='fresh':
                        coarse_noise=torch.randn_like(fine);state=RDDFresh(ratio)(state,clean=clean,alpha=alpha,sigma=sigma,noise=coarse_noise).state
                    elif self.boundary=='bilinear':
                        lifted=torch.nn.functional.interpolate(clean,scale_factor=ratio,mode='bilinear',align_corners=False)
                        coarse_noise=torch.randn_like(lifted);state=alpha*lifted+sigma*coarse_noise
                    elif self.boundary=='noise_filling':
                        previous=self.diffusions[index+1]
                        if previous.timestep_map[0]!=previous.T_c:raise ValueError('Noise filling requires a zero-drift previous endpoint')
                        old_alpha=torch.tensor(previous.sqrt_alphas_cumprod[0],device=initial.device,dtype=torch.float32)
                        old_sigma=torch.tensor(previous.sqrt_one_minus_alphas_cumprod[0],device=initial.device,dtype=torch.float32)
                        recovered=(state[:n].float()-old_alpha*clean)/old_sigma
                        innovation=torch.randn_like(fine)
                        low=recovered.repeat_interleave(ratio,2).repeat_interleave(ratio,3)/float(ratio)
                        h,w=clean.shape[-2:]
                        pooled=innovation.view(n,initial.shape[1],h,ratio,w,ratio).mean(dim=(3,5))
                        projected=pooled.repeat_interleave(ratio,2).repeat_interleave(ratio,3)
                        coarse_noise=low+(innovation-projected)
                        if self.sigma_h==0.:
                            state=RDDFresh(ratio)(state,clean=clean,alpha=alpha,sigma=sigma,noise=coarse_noise).state
                        else:
                            null=coarse_noise-low;null_std=(1-alpha**2+alpha**2*self.sigma_h**2).sqrt()
                            lifted=clean.repeat_interleave(ratio,2).repeat_interleave(ratio,3)
                            state=alpha*lifted+sigma*low+null_std*null
                            coarse_noise=low+(null_std/sigma)*null
                    else:
                        innovation=torch.randn_like(fine)
                        # Source innovation-preserving lift has rho=1, no low draw.
                        from .noise import coupled_lift
                        low=coarse_noise.repeat_interleave(ratio,2).repeat_interleave(ratio,3)/float(ratio)
                        coarse_noise=coupled_lift(coarse_noise,torch.zeros_like(coarse_noise),innovation,ratio,1.)
                        if self.sigma_h==0.:
                            state=RDDFresh(ratio)(state,clean=clean,alpha=alpha,sigma=sigma,noise=coarse_noise).state
                        else:
                            null=coarse_noise-low;null_std=(1-alpha**2+alpha**2*self.sigma_h**2).sqrt()
                            lifted=clean.repeat_interleave(ratio,2).repeat_interleave(ratio,3)
                            state=alpha*lifted+sigma*low+null_std*null
                            coarse_noise=low+(null_std/sigma)*null
                    if self.cfg>1:state=torch.cat([state,state])
                    trace.append(state[:n].clone())
                common=dict(noise=state,clip_denoised=self.clip,model_kwargs=kwargs,device=state.device,progress=False)
                if self.solver.startswith('dpm'):out=diffusion.dpm_solver_sample_loop(forward,state.shape,order=int(self.solver[-1]),**common)
                elif self.solver=='ddim':out=diffusion.ddim_sample_loop(forward,state.shape,eta=0.,**common)
                elif self.solver in ('jump','ddpm'):out=diffusion.jump_step_sample_loop(forward,state.shape,deterministic=False,**common)
                else:out=diffusion.p_sample_loop(forward,state.shape,**common)
                state=out['sample'];clean=out['pred_xstart'][:n].float();trace.append(state[:n].clone())
        state=state[:n].float()
        return (state,trace) if return_trajectory else state
