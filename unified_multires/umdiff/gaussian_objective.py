"""ADM/RDD epsilon, auxiliary drift and learned-range variance objectives.

The variational objective follows OpenAI improved-diffusion/ADM and preserves
its detached mean head and discretized 8-bit likelihood branch at t=0.
"""
import math
import numpy as np
import torch
from .registry import register,build
from .batches import Prepared
from .solvers import DDPM
from .tree import mean_flat

def clean_from_effective_eps(path,state,t,prediction):
    return path.extract(np.sqrt(1/path.abar),t,state)*state-path.extract(np.sqrt(1/path.abar-1),t,state)*prediction

def learned_logvar(path,t,state,values):
    previous=np.append(1.,path.abar[:-1]);variance=path.betas*(1-previous)/(1-path.abar)
    clipped=np.log(np.append(variance[1],variance[1:]))
    minimum=path.extract(clipped,t,state);maximum=path.extract(np.log(path.betas),t,state)
    fraction=(values+1)/2
    return fraction*maximum+(1-fraction)*minimum

def gaussian_vb(path,clean,state,t,eps_prediction,var_values):
    true_mean,true_logvar=DDPM(False).posterior(path,clean,state,t)
    predicted_clean=clean_from_effective_eps(path,state,t,eps_prediction.detach())
    predicted_mean,_=DDPM(False).posterior(path,predicted_clean,state,t)
    logvar=learned_logvar(path,t,state,var_values)
    kl=.5*(-1+logvar-true_logvar+torch.exp(true_logvar-logvar)+(true_mean-predicted_mean)**2*torch.exp(-logvar))
    kl=mean_flat(kl)/np.log(2.)
    inv=torch.exp(-.5*logvar);centered=clean-predicted_mean
    cdf=lambda x:.5*(1+torch.tanh(np.sqrt(2/np.pi)*(x+.044715*torch.pow(x,3))))
    plus=cdf(inv*(centered+1/255));minus=cdf(inv*(centered-1/255))
    log_probs=torch.where(clean<-.999,plus.clamp_min(1e-12).log(),torch.where(clean>.999,(1-minus).clamp_min(1e-12).log(),(plus-minus).clamp_min(1e-12).log()))
    nll=-mean_flat(log_probs)/np.log(2.)
    return torch.where(t==0,nll,kl)

class GaussianPrepared(Prepared):
    def loss(self,prediction):
        features=None
        if self.metadata.get('repa_weight'):
            if not isinstance(prediction,tuple) or len(prediction)!=2:raise ValueError('RDD alignment requires a prediction/features pair')
            prediction,features=prediction
        c=self.state.shape[1];meta=self.metadata;heads=prediction.split(c,dim=1)
        expected=1+int(meta['learned'])+int(meta['decoupled'])
        if len(heads)!=expected or any(h.shape!=self.state.shape for h in heads):raise ValueError('Gaussian head count or shape mismatch')
        terms={'mse':mean_flat((self.target-heads[0])**2)};loss=terms['mse']
        if meta['learned']:
            terms['vb']=gaussian_vb(meta['path'],meta['clean'],self.state,self.time,heads[0],heads[1])*meta['vb_scale'];loss=loss+terms['vb']
        if meta['decoupled']:
            terms['mse_d']=mean_flat((heads[-1]-meta['drift'])**2);loss=loss+terms['mse_d']
        if meta.get('loss_observer') is not None:meta['loss_observer'](self.time,loss.detach())
        scheme=meta.get('loss_weighting','none')
        if scheme!='none':
            alpha=torch.as_tensor(meta['path'].abar,device=self.time.device,dtype=torch.float32)[self.time.long()]
            snr=alpha/(1-alpha).clamp_min(1e-12)
            weights=torch.minimum(snr,torch.full_like(snr,meta['min_snr_gamma']))/snr.clamp_min(1e-12) if scheme=='min-snr' else (meta['p2_k']+snr).pow(-meta['p2_gamma'])
            loss=terms['mse']*weights.to(terms['mse'].dtype)
            if 'vb' in terms:loss=loss+terms['vb']
            if 'mse_d' in terms and not meta['omit_drift_for_native_weighting']:loss=loss+terms['mse_d']
        if meta.get('importance_weights') is not None:loss=loss*meta['importance_weights'].to(device=loss.device,dtype=loss.dtype)
        if features is not None:
            import torch.nn.functional as F
            teacher=meta['teacher_features'];source_side=math.isqrt(teacher.shape[1]);target_side=math.isqrt(features.shape[1])
            if source_side**2!=teacher.shape[1] or target_side**2!=features.shape[1]:raise ValueError('RDD alignment requires square token grids')
            teacher=F.adaptive_avg_pool2d(teacher.transpose(1,2).reshape(len(teacher),teacher.shape[-1],source_side,source_side),(target_side,target_side)).flatten(2).transpose(1,2)
            if teacher.shape!=features.shape:raise ValueError('Student/teacher feature shapes differ')
            terms['repa']=-(F.normalize(features.float(),dim=-1)*F.normalize(teacher.float(),dim=-1)).sum(-1).mean(-1)
            loss=loss.mean()+meta['repa_weight']*terms['repa'].mean()
        return loss,terms

@register('batch','rdd_ddpm')
class GaussianBatch:
    def __init__(self,path=None,learned_variance=False,decoupled=False,rescaled_mse=False,loss_weighting='none',min_snr_gamma=5.,p2_gamma=1.,p2_k=1.,repa_weight=0.,omit_drift_for_native_weighting=False):
        self.path=build('path',{'name':'discrete_vp','args':path or {}});self.learned=learned_variance;self.decoupled=decoupled;self.rescaled=rescaled_mse
        if len(self.path.betas)<2:raise ValueError('Learned posterior table requires at least 2 steps')
        if decoupled and self.path.drift_start is None:raise ValueError('RDD decoupled drift head requires a drift path')
        if loss_weighting not in ('none','min-snr','p2') or min_snr_gamma<=0 or p2_gamma<0 or p2_k<=0 or repa_weight<0:raise ValueError('Invalid optional RDD objective settings')
        self.extra=dict(loss_weighting=loss_weighting,min_snr_gamma=min_snr_gamma,p2_gamma=p2_gamma,p2_k=p2_k,repa_weight=repa_weight,omit_drift_for_native_weighting=omit_drift_for_native_weighting)
    def __call__(self,images,t,stream,noise=None,context=None):
        context=context or {}
        p=self.path;noise=stream.normal(images) if noise is None else noise;sample=p.sample(images,t,noise)
        target=noise
        if p.drift_start is not None:
            mapped=torch.as_tensor(p.timestep_map,device=t.device,dtype=t.dtype)[t.long()]
            fraction=((mapped-p.drift_start)/(p.drift_end-p.drift_start)).reshape(-1,1,1,1)
            abar=p.extract(p.abar,t,images)
            target=noise+p.multiplier*torch.sqrt(abar/(1-abar))*fraction*sample.drift
        network_time=torch.as_tensor(p.timestep_map,device=t.device,dtype=t.dtype)[t.long()]
        extra=dict(self.extra)
        if extra['repa_weight']:
            if 'teacher_features' not in context or not torch.is_tensor(context['teacher_features']):raise ValueError('RDD alignment requires explicitly supplied local teacher_features in context')
            extra['teacher_features']=context['teacher_features']
        for key in ('importance_weights','loss_observer'):
            if key in context:extra[key]=context[key]
        return GaussianPrepared(sample.state,target,t,network_time=network_time,model_kwargs={'return_features':True} if extra['repa_weight'] else {},metadata={'path':p,'clean':images,'drift':sample.drift,'learned':self.learned,'decoupled':self.decoupled,'vb_scale':len(p.betas)/1000 if self.rescaled else 1.,'handles_importance_weights':True,**extra})

@register('sampler','rdd_ddpm')
class GaussianSampler:
    def __init__(self,path=None,learned_variance=False,decoupled=False,clip_clean=True):
        self.path=build('path',{'name':'discrete_vp','args':path or {}});self.learned=learned_variance;self.decoupled=decoupled;self.clip=clip_clean
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        state=initial;trace=[state];p=self.path
        for index in range(len(p.betas)-1,-1,-1):
            t=torch.full((len(state),),index,device=state.device,dtype=torch.long)
            network_time=torch.as_tensor(p.timestep_map,device=t.device,dtype=t.dtype)[t]
            prediction=model(state,network_time,**(model_kwargs or {}));heads=prediction.split(state.shape[1],dim=1)
            if len(heads)!=1+int(self.learned)+int(self.decoupled):raise ValueError('Gaussian inference heads mismatch')
            clean=clean_from_effective_eps(p,state,t,heads[0]);clean=clean.clamp(-1,1) if self.clip else clean
            mean,logvar=DDPM(False).posterior(p,clean,state,t)
            if self.learned:logvar=learned_logvar(p,t,state,heads[1])
            noise=stream.normal(state)
            state=mean+(t!=0).float().reshape(-1,1,1,1)*torch.exp(.5*logvar)*noise;trace.append(state)
        return (state,trace) if return_trajectory else state
