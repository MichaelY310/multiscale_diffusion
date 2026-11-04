"""Paper orchestration with explicit clocks, conditioning, and solver reuse."""
import math
import torch
import torch.nn.functional as F
from .registry import register
from .tree import first

@register('sampler','dctdiff_euler_maruyama')
class DCTEulerMaruyama:
    """Released reverse SDE / probability-flow ODE, including terminal step."""
    def __init__(self,steps=100,snr_scale=1.,stochastic=False,eps=1e-3):self.steps=steps;self.scale=snr_scale;self.stochastic=stochastic;self.eps=eps
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        from . import native
        source=native.load('dctdiff/sde.py');score=source.ScoreModel(model,'noise_pred',source.VPSDE(SNR_scale=self.scale))
        process=(source.ReverseSDE if self.stochastic else source.ODE)(score);trace=[] if return_trajectory else None
        # Native ODE consumes Gaussian draws even with zero diffusion. Preserve
        # those draws instead of silently changing subsequent stream positions.
        with stream.native_rng(initial.device):value=source.euler_maruyama(process,initial,self.steps,eps=self.eps,trace=trace,**(model_kwargs or {}))
        return (value,trace) if return_trajectory else value

@register('sampler','dctdiff_dpm')
class DCTDPMSampler:
    """Author DPM-Solver backend; representation/path/loss are independent ports."""
    def __init__(self,steps=12,snr_scale=1.,eps=1e-4):self.steps=steps;self.scale=snr_scale;self.eps=eps
    @torch.no_grad()
    def __call__(self,model,initial,stream=None,model_kwargs=None,return_trajectory=False):
        if return_trajectory:raise ValueError('Author DCT DPM backend returns final state only')
        from . import native
        source=native.load('dctdiff/dpm_solver_pytorch.py')
        schedule=source.NoiseScheduleVP(schedule='linear',SNR_scale=self.scale)
        def noise_pred(state,t,**kwargs):return model(state,t*999,**kwargs)
        fn=source.model_wrapper(noise_pred,schedule,time_input_type='0',model_kwargs=model_kwargs or {})
        return source.DPM_Solver(fn,schedule).sample(initial,steps=self.steps,eps=self.eps,adaptive_step_size=False,fast_version=True)

@register('sampler','dimr_dpm')
class DiMRDPMSampler:
    def __init__(self,betas,steps=12,guidance_scale=0.,guidance_power=None,null_label=1000,scaling_factor=-1):
        self.betas=betas;self.steps=steps;self.cfg=guidance_scale;self.power=guidance_power;self.null=null_label;self.scaling=scaling_factor
        if guidance_power is not None and not ((guidance_power==-1 and 0<guidance_scale<1) or (guidance_power>0 and guidance_scale>=1)):raise ValueError('Invalid native DiMR guidance scale/power pair')
    @torch.no_grad()
    def __call__(self,model,initial,stream=None,model_kwargs=None,return_trajectory=False):
        if return_trajectory:raise ValueError('Author DiMR DPM backend returns final state only')
        from . import native
        source=native.load('dimr/dpm_solver.py')
        schedule=source.NoiseScheduleVP(schedule='discrete',betas=torch.tensor(self.betas,device=initial.device).float())
        def fn(state,t,scaling_factor=-1):
            if scaling_factor!=-1:state=state/state.std(axis=(1,2,3),keepdims=True)
            time=t*len(self.betas);net_t=time.float()/1000*8-4;kwargs=dict(model_kwargs or {})
            cond=model(state,net_t,**kwargs)[-1]
            if self.power is None:return cond
            kwargs['y']=torch.full((len(state),),self.null,device=state.device,dtype=torch.long)
            uncond=model(state,net_t,**kwargs)[-1]
            if self.power==-1:return cond+self.cfg*(cond-uncond)
            scale=(self.cfg-1)*(1-torch.cos(((1-time[0]/1000)**self.power)*math.pi))*.5+1
            return uncond+scale*(cond-uncond)
        return source.DPM_Solver(fn,schedule,predict_x0=True,thresholding=False,scaling_factor=self.scaling).sample(initial,steps=self.steps,eps=1/len(self.betas),T=1.)

@register('sampler','ppflow')
class PPFlowSampler:
    def __init__(self,steps=12,cfg_scales=None,null_label=1000):self.steps=steps;self.cfg=cfg_scales;self.null=null_label
    @torch.no_grad()
    def __call__(self,model,initial,stream=None,model_kwargs=None,return_trajectory=False):
        kwargs=dict(model_kwargs or {});cfg=kwargs.pop('cfg_scales',self.cfg)
        state=initial;grid=torch.linspace(0,1,self.steps+1,device=initial.device)
        net=model.model if hasattr(model,'model') else model
        if cfg is not None:
            if len(cfg)!=net.patch_level:raise ValueError('One PPFlow CFG scale per stage is required')
            labels=kwargs.get('y')
            if labels is None or len(labels)!=len(initial):raise ValueError('PPFlow guidance needs one conditional label per sample')
            kwargs['y']=torch.cat([labels,torch.full_like(labels,self.null)])
            kwargs['cfg_scales']=cfg;state=torch.cat([initial,initial])
        trace=[initial.clone()]
        for t0,t1 in zip(grid[:-1],grid[1:]):
            forward=net.forward_wo_cfg if cfg is None else net.forward_with_cfg
            state=state+(t1-t0)*forward(state,torch.ones(len(state),device=state.device)*t0,**kwargs)
            trace.append((state if cfg is None else state[:len(initial)]).clone())
        if cfg is not None:state=state[:len(initial)]
        return (state,trace) if return_trajectory else state

@register('sampler','ppflow_native')
class PPFlowNativeSampler:
    """Published ODE/SDE solver policies, using the unchanged author backend.

    points has the source num_sampling_steps meaning (saved endpoints), unlike
    the independent Euler component's steps meaning (updates). Kernel/NFE
    differences are therefore explicit rather than hidden by an off-by-one.
    """
    def __init__(self,points=250,mode='ode',sampling_method=None,cfg_scales=None,
                 null_label=1000,atol=1e-6,rtol=1e-3,reverse=False,sample_eps=0.,**sde_options):
        if points<2 or mode not in ('ode','sde'):raise ValueError('Invalid native PPFlow sampling policy')
        if mode=='ode' and sde_options:raise ValueError('SDE options cannot be used for ODE')
        if mode=='sde':sde_options.setdefault('diffusion_form','sigma')
        if mode=='sde' and sde_options['diffusion_form']=='SBDM' and sample_eps<=0:raise ValueError('Native PPFlow SBDM at t=0 is singular; specify positive sample_eps or the released CLI sigma diffusion')
        self.points=points;self.mode=mode;self.method=sampling_method or ('heun2' if mode=='ode' else 'Euler');self.cfg=cfg_scales
        self.null=null_label;self.atol=atol;self.rtol=rtol;self.reverse=reverse;self.options=sde_options
        self.sample_eps=sample_eps
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        from .native_models import activate
        activate('ppflow',['models','transport','modules'])
        from transport.transport import Transport,ModelType,PathType,WeightType,Sampler
        transport=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,
                            loss_type=WeightType.NONE,train_eps=0.,sample_eps=self.sample_eps)
        sampler=Sampler(transport)
        kwargs=dict(model_kwargs or {});cfg=kwargs.pop('cfg_scales',self.cfg)
        net=model.model if hasattr(model,'model') else model
        state=initial
        if cfg is not None:
            labels=kwargs.get('y')
            if len(cfg)!=net.patch_level or labels is None or len(labels)!=len(initial):raise ValueError('Native PPFlow guidance needs matched stage scales and labels')
            kwargs.update(y=torch.cat([labels,torch.full_like(labels,self.null)]),cfg_scales=cfg)
            state=torch.cat([initial,initial])
        forward=net.forward_wo_cfg if cfg is None else net.forward_with_cfg
        if self.mode=='ode':
            function=sampler.sample_ode(sampling_method=self.method,num_steps=self.points,
                                        atol=self.atol,rtol=self.rtol,reverse=self.reverse)
        else:
            function=sampler.sample_sde(sampling_method=self.method,num_steps=self.points,**self.options)
        with stream.native_rng(initial.device):trajectory=function(state,forward,**kwargs)
        trace=[value if cfg is None else value[:len(initial)] for value in trajectory]
        return (trace[-1],trace) if return_trajectory else trace[-1]

@register('sampler','pixelflow')
class PixelFlowSampler:
    def __init__(self,stages=4,steps=8,train_steps=1000,shift=1.,gamma=-1/3,guidance_scale=0.,solver='euler'):
        from .clocks import PixelFlowClock
        self.clock=PixelFlowClock(train_steps,stages,gamma);self.stages=stages;self.steps=[steps]*stages if isinstance(steps,int) else steps;self.shift=shift;self.gamma=gamma;self.cfg=guidance_scale
        if len(self.steps)!=stages:raise ValueError('Steps must match stages')
        if solver not in ('euler','native_dopri5'):raise ValueError('Unknown PixelFlow solver')
        if solver=='native_dopri5' and (guidance_scale<=0 or len(set(self.steps))!=1):raise ValueError('Released Dopri5 requires CFG and uses the first stage point count for every stage')
        self.solver=solver
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        from .position import rotary_2d
        from .noise import equicorrelated_blocks
        kwargs=dict(model_kwargs or {});labels=kwargs.pop('class_labels',None)
        if labels is None:raise ValueError('Class-conditional PixelFlow requires class_labels')
        if kwargs:raise ValueError('This native PixelFlow preset is class-conditional; register a text-conditioned sampler for text inputs')
        net=model.model if hasattr(model,'model') else model
        state=initial;trace=[state.clone()]
        # Original condition is scale>0, including scale=1 (redundant CFG pass).
        cfg=self.cfg>0
        if cfg:labels=torch.cat([1000*torch.ones_like(labels),labels])
        for stage in range(self.stages):
            grid,network=self.clock.grid(self.steps[stage],stage,self.shift,state.device)
            if stage:
                state=F.interpolate(state,scale_factor=2,mode='nearest')
                original=self.clock.original[stage]
                a=1/(math.sqrt(1-1/self.gamma)*(1-original)+original);b=a*(1-original)/math.sqrt(-self.gamma)
                # Source samples block noise on CPU, then transfers it.
                noise=equicorrelated_blocks(torch.empty_like(state,device='cpu'),self.gamma,stream=stream).to(state)
                state=a*state+b*noise;trace.append(state.clone())
            size=state.shape[-1]//net.patch_size
            rope=torch.stack(rotary_2d(net.attention_head_dim,((0,0),(size,size)),(size,size)),-1).to(state.device)
            if self.solver=='native_dopri5':
                # Preserve the author's doubled state across stage boundaries,
                # including block-noise draws for both halves. Collapsing it
                # after each stage changes subsequent noise and the pixels.
                from .native_models import activate
                activate('pixelflow',['pixelflow'])
                from pixelflow.solver_ode_wrapper import ODE
                ode=ODE(t0=0,t1=1,sampler_type='dopri5',num_steps=self.steps[0],atol=1e-6,rtol=.001)
                if stage==0:state=torch.cat([state]*2)
                scale=(self.cfg-1)*{0:0,1:1/6,2:2/3,3:1}[stage]+1
                trajectory=ode.sample(state,net.c2i_forward_cfg_torchdiffq,
                    self.clock.network_times[stage][0].item(),self.clock.network_times[stage][-1].item(),
                    class_labels=labels,cfg_scale=scale,latent_size=torch.tensor([size],device=state.device,dtype=torch.int32),pos_embed=rope)
                state=trajectory[-1]
                if stage==self.stages-1:state=state[:len(state)//2]
                trace.append(state.clone())
                continue
            for i,clock_t in enumerate(network):
                inputs=torch.cat([state]*2) if cfg else state
                prediction=net(inputs,timestep=clock_t.expand(len(inputs)).to(inputs.dtype),class_labels=labels,latent_size=torch.tensor([size],device=state.device,dtype=torch.int32),pos_embed=rope)
                if cfg:
                    negative,positive=prediction.chunk(2);scale=(self.cfg-1)*{0:0,1:1/6,2:2/3,3:1}[stage]+1
                    prediction=negative+scale*(positive-negative)
                state=(state.float()+(grid[i+1].float()-grid[i].float())*prediction).to(prediction.dtype);trace.append(state.clone())
        return (state,trace) if return_trajectory else state

@register('sampler','lapflow')
class LapFlowSampler:
    def __init__(self,levels=2,activations=None,step_size=.1,solver='euler',curve='linear',atol=1e-5,rtol=1e-5,perturb=False,guidance_scale=1.,null_label=1000,noise_var_factor=None,variance_aware=False,square_policy='reject'):
        self.levels=levels;self.activations=activations or ([.4,0.] if levels==2 else [.6,.3,0.]);self.dt=step_size;self.solver=solver;self.curve=curve;self.atol=atol;self.rtol=rtol;self.perturb=perturb
        self.cfg=guidance_scale;self.null=null_label;self.noise_factor=noise_var_factor;self.variance=variance_aware
        if square_policy not in ('reject','source_bug','corrected'):raise ValueError('Unknown square sampler compatibility policy')
        self.square_policy=square_policy
        if curve=='square' and square_policy=='reject':raise ValueError('Released square sampling swaps noise/time arguments; choose explicit source_bug or corrected policy')
        if levels not in (2,3) or (levels==3 and curve!='linear'):raise ValueError('Released LapFlow supports 2/3 levels, cosine only for 2')
        if levels==3 and (variance_aware or noise_var_factor is not None):raise ValueError('Released three-level LapFlow has no variance-aware or scaled-noise sampling path')
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        from torchdiffeq import odeint
        from .representations import Laplacian
        from .operators import resize
        if self.noise_factor is not None:initial=initial*self.noise_factor
        rep=Laplacian(self.levels,up='nearest',order='fine_first');noises=rep.encode(initial)
        if self.levels==3:
            # The native CFG caller passes a doubled batch before drawing the
            # two lower-resolution noises. forward_with_cfg uses the first
            # half, but both draws still advance the original RNG.
            noises=[initial]+[stream.normal(torch.cat([x,x]))[:len(initial)] if self.cfg>1 else stream.normal(x) for x in noises[1:]]
        state=(noises[-1],);trace=[]
        options={'step_size':self.dt,'perturb':self.perturb} if self.solver in ('euler','midpoint','rk4','explicit_adams','implicit_adams') else {'dtype':torch.float64}
        kwargs=dict(model_kwargs or {});net=model.model if hasattr(model,'model') else model
        if self.cfg>1:
            labels=kwargs.get('y')
            if labels is None:raise ValueError('LapFlow CFG requires labels')
            kwargs['y']=torch.cat([labels,torch.full_like(labels,self.null)]);kwargs['cfg_scale']=self.cfg
        def integrate(bands,start,end,extra_tail=False):
            def fn(t,x):
                # MoT's source inference explicitly disables variance-aware scaling.
                var=None if getattr(net,'use_mot',False) or not self.variance else [e.std() for e in noises[-len(x):]]
                if self.cfg>1:
                    doubled=tuple(torch.cat([v,v]) for v in x)
                    return tuple(v[:len(x[0])] for v in net.forward_with_cfg(t,doubled,var_scale=var,**kwargs))
                # The native non-CFG caller sends the scalar ODE time. Expanding
                # it to batch changes the time-MLP GEMM shape and FP32 rounding.
                return tuple(model(x,t,**({**kwargs,'var_scale':var} if self.variance else kwargs)))
            times=[start,end,1.] if extra_tail else [start,end]
            result=odeint(fn,bands,torch.tensor(times,device=initial.device,dtype=initial.dtype),method=self.solver,options=options,atol=self.atol,rtol=self.rtol)
            return tuple(x[1] if extra_tail else x[-1] for x in result)
        for band in range(self.levels-1,-1,-1):
            start=self.activations[band];end=self.activations[band-1] if band else 1.
            if band<self.levels-1:
                if self.curve=='square':
                    if self.square_policy=='source_bug':innovation=torch.sin((1-noises[band])**2*math.pi/2)*start
                    else:innovation=torch.sin(torch.as_tensor(start,device=initial.device,dtype=initial.dtype).neg().add(1).square()*math.pi/2)*noises[band]
                else:
                    sigma=1-start if self.curve=='linear' else math.sin((1-start)*math.pi/2)
                    innovation=noises[band]*sigma
                state=(innovation,*state)
            # Native 2-level code integrates an unused coarse tail to t=1.
            state=integrate(state,start,end,extra_tail=self.levels==2 and band==1);trace.append(state)
        # Source three-level reconstruction is left-associated. The generic
        # recursive Laplacian inverse is mathematically identical but rounds
        # differently after summing the three FP32 bands.
        image=state[0]+resize(state[1],scale=2,mode='nearest')+resize(resize(state[2],scale=2,mode='nearest'),scale=2,mode='nearest') if self.levels==3 else rep.decode(state)
        return (image,trace) if return_trajectory else image

@register('sampler','mdm')
class MDMSampler:
    def __init__(self,gammas,scales=(2,1),schedule_shifted=True,power=1.,quantity='v',eta=None,guidance_scale=1.,threshold='legacy',resample_steps=False,inference_steps=None,clip_output=False):
        self.gammas=torch.tensor(gammas,dtype=torch.float32);self.scales=scales;self.shift=schedule_shifted;self.power=power;self.quantity=quantity;self.eta=eta
        self.cfg=guidance_scale;self.threshold=threshold;self.resample=resample_steps;self.steps=inference_steps;self.clip_output=clip_output
        if resample_steps and (inference_steps is None or not 1<=inference_steps<len(gammas)):raise ValueError('MDM resampling requires 1..training_steps inference steps')
        if not resample_steps and inference_steps is not None:raise ValueError('MDM inference_steps requires resample_steps=True')
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        from .multiscale import snr_shift,mdm_posterior
        from .operators import block_average
        state=list(initial) if isinstance(initial,(tuple,list)) else [initial]+[stream.normal(block_average(initial,self.scales[0]//s) if initial.ndim==5 else F.avg_pool2d(initial,self.scales[0]//s)) for s in self.scales[1:]]
        trace=[state];gammas=self.gammas.to(state[0].device)
        if self.resample:
            # The source rounds an (N+1)/(K+1) grid: for K<N its first
            # index can be below N. Preserve that endpoint rather than
            # substituting a conventional N..0 linspace.
            import numpy as np
            indices=(np.arange(self.steps+1)*(len(gammas)/(self.steps+1))).round()[::-1].copy().astype(np.int64).tolist()
        else:indices=list(range(len(gammas)-1,-1,-1))
        for index,previous in zip(indices[:-1],indices[1:]):
            t=torch.full((len(state[0]),),index-1,device=state[0].device,dtype=torch.long)
            kwargs=dict(model_kwargs or {});unconditional=kwargs.pop('unconditional',None)
            if self.cfg!=1:
                if unconditional is None or set(unconditional)!=set(kwargs):raise ValueError('MDM CFG requires matching conditional and unconditional fields')
                both={k:torch.cat([unconditional[k],v]) if torch.is_tensor(v) else v for k,v in kwargs.items()}
                prediction=model([torch.cat([x,x]) for x in state],torch.cat([t,t]),**both)
                prediction=[u+self.cfg*(c-u) for u,c in (p.chunk(2) for p in prediction)]
            else:prediction=model(state,t,**kwargs)
            updates=[]
            for x,p,s in zip(state,prediction,self.scales):
                g=gammas[index].expand(len(x)).reshape(-1,*([1]*(x.ndim-1)));gp=gammas[previous].expand_as(g)
                if self.shift:g=snr_shift(g,s,self.power);gp=snr_shift(gp,s,self.power)
                noise=stream.normal(x) if index!=1 and self.eta!=0 else None
                updates.append(mdm_posterior(x,p,g,gp,self.quantity,noise,self.eta,1. if self.shift else s,self.threshold)[1])
            state=updates;trace.append(state)
        output=state[0] if self.shift else state[0]*self.scales[0]
        if self.clip_output:output=output.clamp(-1,1)
        return (output,trace) if return_trajectory else output

@register('sampler','nami')
class NAMISampler:
    def __init__(self,steps=(10,10,10),guidance_scale=3.,clock=None,if_use_pool=True,use_mask=False):
        from .registry import build
        self.steps=steps;self.cfg=guidance_scale;self.clock=build('clock',clock or {'name':'nami'});self.pool=if_use_pool;self.mask=use_mask
    @torch.no_grad()
    def __call__(self,model,initial,stream=None,model_kwargs=None,return_trajectory=False):
        kwargs=dict(model_kwargs or {});bridge=kwargs.pop('bridge_flow');text=kwargs.pop('prompt_embeds');pooled=kwargs.pop('pooled_prompt_embeds')
        negative=kwargs.pop('negative_prompt_embeds',None);negative_pooled=kwargs.pop('negative_pooled_prompt_embeds',None)
        text_mask=kwargs.pop('prompt_attention_mask',None);negative_mask=kwargs.pop('negative_attention_mask',None)
        if kwargs:raise ValueError(f'Unknown NAMI conditioning: {set(kwargs)}')
        if self.cfg>1:
            if negative is None or negative_pooled is None:raise ValueError('CFG requires supplied negative embeddings')
            text=torch.cat([negative,text]);pooled=torch.cat([negative_pooled,pooled])
            if self.mask:text_mask=torch.cat([negative_mask,text_mask])
        # Native ignores its latents argument and draws a full final-grid tensor,
        # then downsamples and variance-normalizes. Here initial IS that full draw.
        state=initial
        for _ in range(len(self.steps)-1):state=F.interpolate(state,scale_factor=.5,mode='bilinear')*2
        trace=[state];txt_ids=torch.zeros(text.shape[1],3,device=state.device,dtype=text.dtype)
        for stage,n in enumerate(self.steps):
            times,sigmas=self.clock.grid(n,stage,state.device)
            if stage:
                from .boundaries import LearnedAffine
                result=bridge(state,stage=stage) if isinstance(bridge,LearnedAffine) else bridge(state,stage)
                state=result.state if hasattr(result,'state') else result;trace.append(state)
            for i,t in enumerate(times):
                inputs=torch.cat([state]*2) if self.cfg>1 else state;b,c,h,w=inputs.shape
                packed=inputs.view(b,c,h//2,2,w//2,2).permute(0,2,4,1,3,5).reshape(b,h*w//4,c*4)
                ids=torch.zeros(h//2,w//2,3);ids[...,1]+=torch.arange(h//2)[:,None];ids[...,2]+=torch.arange(w//2)[None,:];ids=ids.reshape(-1,3).to(device=state.device,dtype=text.dtype)
                attention=None
                if self.mask:
                    image_ids=torch.arange(1,len(text_mask)+1,dtype=text_mask.dtype,device=text_mask.device)[:,None].repeat(1,len(ids));mask_ids=torch.cat([text_mask,image_ids],1)
                    attention=mask_ids[:,None,:,None]==mask_ids[:,None,None,:]
                pred=model(hidden_states=packed,timestep=t.expand(b).to(inputs.dtype)/1000,guidance=None,pooled_projections=pooled,encoder_hidden_states=text,txt_ids=txt_ids,img_ids=ids,joint_attention_kwargs=None,return_dict=False,attention_mask=attention,if_use_pool=self.pool,stage_index=stage)[0]
                if self.cfg>1:
                    uncond,cond=pred.chunk(2);pred=uncond+self.cfg*(cond-uncond)
                pred=pred.view(len(state),h//2,w//2,c,2,2).permute(0,3,1,4,2,5).reshape_as(state)
                state=(state.float()+(sigmas[i+1]-sigmas[i])*pred).to(pred.dtype);trace.append(state)
        return (state,trace) if return_trajectory else state

@register('sampler','speed_flux')
class SPEEDFluxSampler:
    def __init__(self,full_hw=(16,16),scales=(.5,1.),transitions=(.5,),steps=12,transform='dct',seed=0,guidance=3.5,scheduler=None):
        self.hw=full_hw;self.scales=scales;self.transitions=transitions;self.steps=steps;self.transform=transform;self.seed=seed;self.guidance=guidance;self.scheduler_config=scheduler or {}
        if len(transitions)!=len(scales)-1 or scales[-1]!=1:raise ValueError('Invalid SPEED stage schedule')
    @torch.no_grad()
    def __call__(self,model,initial,stream=None,model_kwargs=None,return_trajectory=False):
        import numpy as np
        from diffusers import FlowMatchEulerDiscreteScheduler
        from .boundaries import SpectralBoundary
        scheduler=FlowMatchEulerDiscreteScheduler(**self.scheduler_config);cfg=scheduler.config
        seq=(self.hw[0]//2)*(self.hw[1]//2)
        mu=(cfg.max_shift-cfg.base_shift)/(cfg.max_image_seq_len-cfg.base_image_seq_len)*(seq-cfg.base_image_seq_len)+cfg.base_shift
        scheduler.set_timesteps(device=initial.device,sigmas=np.linspace(1,1/self.steps,self.steps).tolist(),mu=mu)
        sigmas=scheduler.sigmas;transition_steps=[next((i for i,s in enumerate(sigmas) if s<=threshold),len(sigmas)-1) for threshold in self.transitions]
        kwargs=dict(model_kwargs or {});text=kwargs.pop('prompt_embeds');pooled=kwargs.pop('pooled_prompt_embeds');txt_ids=kwargs.pop('text_ids')
        if kwargs:raise ValueError(f'Unknown SPEED conditioning: {set(kwargs)}')
        state=initial;trace=[state];guidance=torch.full([len(state)],self.guidance,device=state.device,dtype=torch.float32)
        def dims(scale):return tuple(max(2,round(scale*n)//2*2) for n in self.hw)
        def pack(x):
            b,c,h,w=x.shape;return x.view(b,c,h//2,2,w//2,2).permute(0,2,4,1,3,5).reshape(b,h*w//4,c*4)
        def unpack(x,h,w):return x.view(len(x),h//2,w//2,initial.shape[1],2,2).permute(0,3,1,4,2,5).reshape(len(x),initial.shape[1],h,w)
        packed=pack(state)
        for stage,(start,end,scale) in enumerate(zip([0]+transition_steps,transition_steps+[self.steps],self.scales)):
            h,w=dims(scale);ids=torch.zeros(h//2,w//2,3);ids[...,1]+=torch.arange(h//2)[:,None];ids[...,2]+=torch.arange(w//2)[None,:];ids=ids.reshape(-1,3).to(state)
            for i in range(start,end):
                t=scheduler.timesteps[i];prediction=model(hidden_states=packed,timestep=t.expand(len(packed)).to(packed.dtype)/1000,guidance=guidance,pooled_projections=pooled,encoder_hidden_states=text,txt_ids=txt_ids,img_ids=ids,return_dict=False)[0]
                packed=scheduler.step(prediction,t,packed,return_dict=False)[0];trace.append(unpack(packed,h,w))
            if stage+1<len(self.scales):
                state=unpack(packed,h,w);time=sigmas[end].item() if end<self.steps else 0.
                boundary=SpectralBoundary(self.transform)(state,time=time,target_hw=dims(self.scales[stage+1]),ratio=self.scales[stage+1]/scale,seed=self.seed+(stage+1)*10000)
                scheduler.sigmas[end]=boundary.time;scheduler.timesteps[end]=boundary.time*1000;scheduler._step_index=None;scheduler._begin_index=None;scheduler.set_begin_index(end)
                state=boundary.state;packed=pack(state.to(initial.dtype));trace.append(state)
        state=unpack(packed,*self.hw)
        return (state,trace) if return_trajectory else state
