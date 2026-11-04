"""RMD: RF trajectory components and sequential fake-score/generator updates.

Independent port of the pinned released implementation. The teacher is an
explicit module, never downloaded. Optional source GAN branch is excluded.
"""
from copy import deepcopy
from pathlib import Path
import os,math,random
import numpy as np
import torch
import torch.nn.functional as F
from .registry import register,build
from .randomness import GlobalRandomStream

@register('model','wan_transformer')
def wan_transformer(**kwargs):
    from diffusers import WanTransformer3DModel
    return WanTransformer3DModel(**kwargs)

def take(table,index,state):return table.to(state.device)[index].reshape(len(index),*((1,)*(state.ndim-1)))

@register('clock','rmd')
class RMDClock:
    def __init__(self,shift=3.,high_shift=5.,train_steps=1000):
        if train_steps!=1000:raise ValueError('Released RMD interval sampler hardcodes top timestep980; use train_steps=1000')
        time=torch.from_numpy(np.linspace(1,train_steps,train_steps,dtype=np.float32).copy())/train_steps
        self.sigmas=shift*time/(1+(shift-1)*time)
        self.high_sigmas=high_shift*time/(1+(high_shift-1)*time)

@register('boundary','spatial_resize')
class SpatialResize:
    def __init__(self,mode='bilinear'):self.mode=mode
    def __call__(self,state,*,target_size,**kwargs):
        from .boundaries import BoundaryResult
        if state.ndim==4:return BoundaryResult(F.interpolate(state,size=target_size,mode=self.mode))
        b,c,t,h,w=state.shape
        flat=state.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        out=F.interpolate(flat,size=target_size,mode=self.mode)
        return BoundaryResult(out.view(b,t,c,*target_size).permute(0,2,1,3,4))

@register('boundary','rmd_reinjection')
class Reinjection:
    def __init__(self,eta=.5):
        if not 0<=eta<=1:raise ValueError('eta must be in [0,1]')
        self.eta=eta
    def __call__(self,clean,*,predicted_noise,sigma,stream,fresh=False,**kwargs):
        from .boundaries import BoundaryResult
        innovation=stream.normal(predicted_noise)
        noise=innovation if fresh else self.eta*predicted_noise+math.sqrt(1-self.eta**2)*innovation
        return BoundaryResult((1-sigma)*clean+sigma*noise,metadata={'noise':noise})

class RMDPredictor:
    def __init__(self,clock,negative,stream,dtype=torch.float32,eta=.5,up='bilinear'):
        self.clock=clock;self.negative=negative;self.stream=stream;self.dtype=dtype
        self.resize=SpatialResize(up);self.reinject=Reinjection(eta)
    def sigma(self,index,state,high=False):return take(self.clock.high_sigmas if high else self.clock.sigmas,index,state)
    def predict(self,model,state,index,text,cfg=None,return_double=False,high=False):
        sigma=self.sigma(index,state,high);time=(sigma*1000).squeeze().reshape(len(state)).to(self.dtype)
        v=model(hidden_states=state.to(self.dtype),timestep=time,encoder_hidden_states=text.to(self.dtype),return_dict=False)[0]
        clean=v*(-sigma)+state;noise=state+(1-sigma)*v
        if cfg is not None:
            negative=self.negative.to(state.device,dtype=self.dtype)
            negative=negative.expand(len(state),-1,-1) if len(negative)==1 and len(state)>1 else negative[:len(state)]
            vu=model(hidden_states=state.to(self.dtype),timestep=time,encoder_hidden_states=negative,return_dict=False)[0]
            cu=vu*(-sigma)+state;nu=state+(1-sigma)*vu
            clean=cu+cfg*(clean-cu);noise=nu+cfg*(noise-nu)
        return (noise,clean) if return_double else clean
    def add_noise(self,state,noise,start,end,high=False):
        s=self.sigma(start,state,high);sn=self.sigma(end,state,high)
        value=state/(1-s)*(1-sn);beta=sn**2-(s/(1-s)*(1-sn))**2
        return (value+beta**.5*noise).to(self.dtype)
    def generate(self,model,noise,text,steps=6,transition=1,switch=2,target_size=(8,12),mid_points=None):
        # Consume the native redundant index draw even though its value is fixed.
        index=self.stream.randint(999,1000,(len(noise),),device=noise.device).long()
        zero=torch.zeros_like(index);state=noise;clean_states=[];noisy=[]
        for step in range(steps):
            sigma=self.sigma(index,state,transition==2 and step>switch);noisy.append(state)
            time=(sigma*1000).squeeze().reshape(len(state))
            v=model(hidden_states=state,encoder_hidden_states=text,timestep=time,return_dict=False)[0]
            clean=v*(-sigma)+state;predicted=state+(1-sigma)*v
            if step==switch and transition in (1,2):
                clean=self.resize(clean,target_size=target_size).state
                predicted=self.resize(predicted,target_size=target_size).state
            clean_states.append(clean)
            index=mid_points[step+1]+zero if mid_points is not None else index-1000//steps
            index[index<0]=index[index<0].clone()*0
            next_sigma=self.sigma(index,noise,transition==2 and step>=switch)
            state=self.reinject(clean,predicted_noise=predicted,sigma=next_sigma,stream=self.stream,fresh=step==switch).state.to(self.dtype)
        noisy.append(clean)
        return clean,clean_states,noisy

def score_loss(pred,target,sigma,mode='inverse_sigma'):
    s=sigma.double()
    if mode=='inverse_sigma':weight=((1-s)/(s+1e-8))**2
    elif mode=='legacy_snr':weight=((s/(1-s+1e-8))**2).clamp(max=5.)
    else:raise ValueError('Unknown RMD score weighting')
    return (weight.float()*(pred-target)**2).reshape(len(pred),-1).mean(1).mean()

def revision_loss(clean,real,fake,shape):
    revised=clean+real-fake
    # Preserve released BCTHW reduction: width is intentionally not reduced.
    weight=(clean-real).abs().mean(dim=[1,2,3],keepdim=True).detach().clip(.2)
    dimensions=shape[1:] if len(shape)==5 else shape
    c=1e-3/math.sqrt(64*64*4)*math.sqrt(math.prod(dimensions))
    diff=clean.float()-revised.detach().float()
    return ((torch.sqrt(diff**2+c**2)-c)/weight).mean()

class RMDObjective:
    def __init__(self,negative,stream,dtype=torch.float32,k_step=6,eta=.5,flow_shift=3.,flow_shift_trans=1,low_rs_step=3,warmup_step=15,split_timestep=True,t_offset_factor=.25,timestep_offset_mode='scaled',cfg=5.,score_weighting_mode='inverse_sigma',target_size=(8,12),upsampler_mode='bilinear',infer_upsampler_mode='bilinear'):
        if k_step<2 or not 0<low_rs_step<k_step:raise ValueError('RMD fixture requires a valid low/high split')
        if flow_shift_trans not in (0,1,2):raise ValueError('Unknown RMD transition')
        self.steps=k_step;self.switch=low_rs_step-1;self.mid=k_step-self.switch
        self.scales=[t_offset_factor]*low_rs_step+[t_offset_factor-.15]*(k_step-low_rs_step)
        self.transition=flow_shift_trans;self.warmup=warmup_step;self.split=split_timestep;self.offset=timestep_offset_mode;self.cfg=cfg;self.mode=score_weighting_mode;self.target=target_size
        self.stream=stream;self.clock=RMDClock(flow_shift);self.predictor=RMDPredictor(self.clock,negative,stream,dtype,eta,upsampler_mode)
        self.infer_resize=SpatialResize(infer_upsampler_mode)
    def trajectory(self,model,template,text):
        # Both draws exist in the native train loop, including unused noise_init.
        self.stream.normal(template);new_noise=self.stream.normal(template)
        model.eval()
        original_resize=self.predictor.resize;self.predictor.resize=self.infer_resize
        with torch.no_grad():_,clean,noisy=self.predictor.generate(model,new_noise,text,self.steps,self.transition,self.switch,self.target)
        self.predictor.resize=original_resize;noisy.reverse()
        indices=self.stream.randint(self.mid,self.steps,(len(template),),device=template.device).long()
        example=self.stream.normal(clean[int(indices[0])])
        for i in range(len(template)):example[i]=clean[int(indices[i])][i]
        model.train()
        return example,noisy
    def sample_times(self,batch,device,step):
        if self.transition and (step<self.warmup or self.stream.uniform((1,),device=device).item()<.5):low,high=self.mid,self.steps+1
        elif self.transition:low,high=1,self.mid
        else:low,high=1,self.steps+1
        band=self.stream.randint(low,high,(batch,),device=device).long()
        generation=band*1000//self.steps-1;middle=generation-1000//self.steps+1;times=generation.clone()*0
        for i in range(batch):
            top=980 if not self.split or band[i]>=self.mid else self.mid*1000//self.steps-1
            offset=0 if self.offset=='zero' else int((top-middle[i].item())*self.scales[self.steps-band[i]])
            times[i]=self.stream.randint(offset+middle[i],top,(1,),device=device)[0]
        return band,times.long(),generation,middle
    def phase(self,phase,generator,fake,real,example,trajectory,text,step):
        p=self.predictor;band,t,tg,tm=self.sample_times(len(example),example.device,step)
        state=torch.empty_like(trajectory[int(band[0])])
        for i in range(len(example)):state[i]=trajectory[int(band[i])][i]
        high=self.transition==2 and band[0]<self.mid
        def forward():
            predicted,clean=p.predict(generator,state,tg,text,return_double=True,high=high)
            if band[0]>=self.mid:
                clean=p.resize(clean,target_size=self.target).state;predicted=p.resize(predicted,target_size=self.target).state
            return predicted,clean
        if phase==1:
            with torch.no_grad():predicted,clean=forward()
        else:predicted,clean=forward()
        sigma=p.sigma(tm,example,self.transition==2)
        mid=p.reinject(clean,predicted_noise=predicted,sigma=sigma,stream=self.stream).state
        destruction=p.add_noise(mid.detach(),self.stream.normal(mid),tm,t,high=True if phase==1 else self.transition==2)
        if phase==1:
            output=p.predict(fake,destruction,t,text,high=True)
            return score_loss(output,clean,p.sigma(t,output,self.transition==2),self.mode)
        with torch.no_grad():
            target=p.predict(real,destruction,t,text,cfg=self.cfg,high=self.transition==2)
            estimated=p.predict(fake,destruction,t,text,high=self.transition==2)
        return revision_loss(clean,target,estimated,example.shape)

class RMDTrainer:
    def __init__(self,config,generator,negative,seed=0,initialize=True):
        self.config=config;self.model=generator;self.fake=deepcopy(generator);self.real=deepcopy(generator).eval().requires_grad_(False)
        checkpoints=config.get('checkpoints',{})
        if set(checkpoints)-{'model','fake','real'}:raise ValueError('RMD checkpoints are model, fake and real local state dicts')
        if initialize and checkpoints:
            if set(checkpoints)!={'model','fake','real'}:raise ValueError('Supply all three RMD local checkpoints explicitly')
            for key,path in checkpoints.items():
                if str(path).endswith('.safetensors'):
                    from safetensors.torch import load_file
                    state=load_file(str(path),device='cpu')
                else:state=torch.load(path,map_location='cpu',weights_only=True)
                getattr(self,key).load_state_dict(state,strict=True)
        training=config.get('training',{});self.precision=training.get('precision','float32');self.accumulation=training.get('accumulation_steps',1)
        if self.precision not in ('float32','bfloat16','float16') or self.accumulation<1:raise ValueError('Invalid RMD precision/accumulation recipe')
        self.dtype={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}[self.precision]
        self.stream=GlobalRandomStream();self.objective=RMDObjective(negative,self.stream,dtype=self.dtype,**config.get('distillation',{}));self.step_index=0;self.data_binding=None
        self.scaler=torch.amp.GradScaler(next(generator.parameters()).device.type,enabled=self.precision=='float16')
        settings=config.get('training',{}).get('optimizer',{})
        allowed={'lr','fake_lr','betas','weight_decay','eps','grad_clip'}
        if set(settings)-allowed:raise ValueError('Unknown distillation optimizer setting')
        common=dict(betas=tuple(settings.get('betas',(0.,.95))),weight_decay=settings.get('weight_decay',1e-4),eps=settings.get('eps',1e-8))
        self.optimizer=torch.optim.AdamW(generator.parameters(),lr=settings.get('lr',5e-6),**common)
        self.fake_optimizer=torch.optim.AdamW(self.fake.parameters(),lr=settings.get('fake_lr',2e-5),**common);self.clip=settings.get('grad_clip',1.)
    def step(self,template,text,before_gradient_processing=None):
        self.model.train();self.fake.train();self.optimizer.zero_grad(set_to_none=True);self.fake_optimizer.zero_grad(set_to_none=True)
        if len(template)%self.accumulation:raise ValueError('RMD effective batch must divide accumulation_steps')
        losses=[0.,0.];width=len(template)//self.accumulation
        # Preserve native phase interleaving per microbatch. On the last one,
        # fake-score weights update before that microbatch's generator phase.
        for i in range(self.accumulation):
            local=template[i*width:(i+1)*width].to(self.dtype);condition=text[i*width:(i+1)*width].to(self.dtype);sync=i==self.accumulation-1
            with torch.autocast(template.device.type,dtype=self.dtype if self.precision!='float32' else torch.bfloat16,enabled=self.precision!='float32'):
                example,trajectory=self.objective.trajectory(self.model,local,condition)
            for phase,optimizer,network in [(1,self.fake_optimizer,self.fake),(2,self.optimizer,self.model)]:
                with torch.autocast(template.device.type,dtype=self.dtype if self.precision!='float32' else torch.bfloat16,enabled=self.precision!='float32'):
                    loss=self.objective.phase(phase,self.model,self.fake,self.real,example,trajectory,condition,self.step_index)/self.accumulation
                if not torch.isfinite(loss):raise FloatingPointError('Non-finite RMD loss')
                self.scaler.scale(loss).backward();losses[phase-1]=losses[phase-1]+loss.detach()
                if sync:
                    self.scaler.unscale_(optimizer)
                    if before_gradient_processing is not None:before_gradient_processing(phase,network)
                    for parameter in network.parameters():
                        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():raise FloatingPointError('Non-finite RMD gradient')
                    if phase==2 and self.clip is not None:torch.nn.utils.clip_grad_norm_(network.parameters(),self.clip,error_if_nonfinite=True)
                    self.scaler.step(optimizer);optimizer.zero_grad(set_to_none=True)
        self.scaler.update()
        self.step_index+=1;return losses
    def save(self,path):
        state=dict(format='rmd1',config=self.config,model=self.model.state_dict(),fake=self.fake.state_dict(),real=self.real.state_dict(),optimizer=self.optimizer.state_dict(),fake_optimizer=self.fake_optimizer.state_dict(),step=self.step_index,negative=self.objective.predictor.negative,scaler=self.scaler.state_dict(),data_binding=self.data_binding,torch_rng=torch.get_rng_state(),numpy_rng=np.random.get_state(),python_rng=random.getstate(),cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)
        if hasattr(self,'sample_text'):state['sample_text']=self.sample_text
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_name(path.name+f'.{os.getpid()}.tmp');torch.save(state,temporary);temporary.replace(path)
    def load(self,path):
        state=torch.load(path,map_location=next(self.model.parameters()).device,weights_only=False)
        if state['format']!='rmd1' or state['config']!=self.config:raise ValueError('Distillation checkpoint config mismatch')
        if self.data_binding is not None and state.get('data_binding')!=self.data_binding:raise ValueError('Checkpoint prompt data identity, batch size or traversal seed mismatch')
        self.data_binding=state.get('data_binding')
        for key in ('model','fake','real','optimizer','fake_optimizer'):getattr(self,key).load_state_dict(state[key])
        self.step_index=state['step'];self.objective.predictor.negative=state['negative']
        self.scaler.load_state_dict(state.get('scaler',{}))
        torch.set_rng_state(state['torch_rng'].cpu());np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
        if state['cuda_rng'] is not None:torch.cuda.set_rng_state_all(state['cuda_rng'])
