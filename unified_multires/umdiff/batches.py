"""Batch assembly is a component: packing, clocks and active scales are explicit."""
from dataclasses import dataclass,field
import math
import numpy as np
import torch
from .registry import register,build
from .tree import first,mean_flat
from .operators import rolling_noise,heat_blur,block_average

@dataclass
class Prepared:
    state:object
    target:object
    time:torch.Tensor
    model_kwargs:dict=field(default_factory=dict)
    leaf_weights:object=None
    loss_weight:object=1.
    segment_lengths:object=None
    metadata:dict=field(default_factory=dict)
    reduction:str='sum_leaves'
    network_time:object=None
    leaf_masks:object=None
    leaf_normalizers:object=None
    def loss(self,prediction):
        terms={}
        def one(pred,target,key,weight):
            if pred.shape!=target.shape:raise ValueError(f'Training target shape mismatch: {key} {pred.shape} {target.shape}')
            error=(pred-target)**2
            if self.reduction=='per_pixel':
                value=error*self.loss_weight
            elif self.reduction=='native_global':
                # LapFlow's trainer uses one F.mse_loss reduction over the
                # entire batch, before combining band losses. Per-example
                # means followed by a batch mean have different rounding.
                value=torch.nn.functional.mse_loss(pred,target)*self.loss_weight
            elif self.reduction=='native_packed':
                # PixelFlow reduces each image's entire token/channel block
                # once. Channel mean followed by token mean rounds differently.
                error=(pred.float()-target.float())**2
                value=torch.stack([part.mean() for part in error.split(self.segment_lengths)]).mean()*self.loss_weight
            else:
                reduce=(lambda x:x.mean(dim=list(range(1,x.ndim)))) if self.reduction=='native_dims' else mean_flat
                value=reduce(error*self.loss_weight) if torch.is_tensor(self.loss_weight) and self.loss_weight.ndim>1 else reduce(error)*self.loss_weight
                if self.segment_lengths is not None:value=torch.stack([part.mean() for part in value.split(self.segment_lengths)])
            if self.leaf_normalizers is not None:value=value/self.leaf_normalizers[int(key)]
            if self.leaf_masks is not None:value=value.masked_fill(~self.leaf_masks[int(key)],0.)
            terms[key]=value
            return value*weight
        if self.reduction=='examples':
            if len(prediction)!=len(self.target):raise ValueError('Packed example count mismatch')
            total=torch.stack([(p-q).square().mean() for p,q in zip(prediction,self.target)])
            terms={'mse':total}
        elif isinstance(self.target,(list,tuple,dict)):
            keys=list(self.target) if isinstance(self.target,dict) else list(range(len(self.target)))
            weights=self.leaf_weights or [1.]*len(keys)
            if len(weights)!=len(keys):raise ValueError('Loss leaf weights mismatch')
            total=sum(one(prediction[k],self.target[k],str(k),w) for k,w in zip(keys,weights))
        else:total=one(prediction,self.target,'mse',1.)
        return total,terms

@dataclass
class DFMPrepared(Prepared):
    native_debug_rng:bool=False
    def loss(self,prediction):
        # Author logging gates consume NumPy draws and shift the NEXT stage
        # choice. This optional native recipe policy preserves those draws;
        # it is not a diffusion/path component or an altered objective.
        if self.native_debug_rng:
            for key,weight in zip(self.target,self.leaf_weights):
                if weight>0 and torch.isfinite(prediction[key]).all():np.random.rand()
        value,terms=super().loss(prediction)
        if self.native_debug_rng:
            for key,weight in zip(self.target,self.leaf_weights):
                if weight>0 and torch.isfinite(terms[str(key)]).all() and torch.isfinite(terms[str(key)]*weight).all():np.random.rand()
        return value,terms

@dataclass
class REPAFeaturesPrepared(DFMPrepared):
    teacher_features:object=None
    repa_weight:float=1.
    def loss(self,prediction):
        if not isinstance(prediction,tuple) or len(prediction)!=2:raise ValueError('DFM REPA requires (predictions, projected_features) from the model')
        denoising,features=prediction
        if features is None:raise ValueError('Enable the native model REPA projectors')
        loss,terms=super().loss(denoising)
        from .regularizers import NativeDFMREPA
        auxiliary=NativeDFMREPA(self.repa_weight)(self.teacher_features,features)
        terms['repa']=auxiliary
        return loss.mean()+auxiliary,terms

@register('batch','relay')
class RelayBatch:
    def __init__(self,patch=4,p_mean=-1.2,p_std=1.2,sigma_data=.5,prob_length=.93,blur_max=3.,min_scale=.001,block_scale=0.,truncate_sigma=.0075,augmentation=None,native_loss=False):
        self.patch=patch;self.mean=p_mean;self.std=p_std;self.data=sigma_data;self.length=prob_length;self.blur=blur_max;self.floor=min_scale;self.block=block_scale;self.truncate=truncate_sigma
        self.augment=relay_augmentation(augmentation)
        self.native_loss=native_loss
    def __call__(self,images,t,stream,noise=None,context=None):
        import scipy.stats as stats
        if images.ndim==5 and self.augment is not None:raise ValueError('Video Relay requires an explicit temporally consistent augmentation adapter')
        u=t.detach().cpu().float().reshape(-1,*([1]*(images.ndim-1)))
        minimum=stats.norm.cdf((np.log(self.truncate)-self.mean)/self.std)*self.length
        u=u.clamp(min=minimum);blur=self.blur*torch.sin(u*torch.pi/2)**2
        normal=torch.tensor(stats.norm.ppf(u*self.length),device=images.device)
        sigma=(normal*self.std+self.mean).exp()
        weight=((sigma**2+self.data**2)/(sigma*self.data)**2).clamp(max=100000)
        images,augment_labels=self.augment(images) if self.augment is not None else (images,None)
        blurred=heat_blur(images,blur.to(images.device),self.patch,self.floor)
        if noise is None:
            e0=stream.normal(blurred)
            draws=[e0] if self.block<=0 else [e0,stream.normal(blurred),stream.normal(blurred)]
        else:draws=noise if isinstance(noise,list) else [noise]
        n=draws[0]*sigma
        if self.block>0:
            if len(draws)!=3:raise ValueError('Relay block noise requires all three raw draws, including discarded first draw')
            n=draws[1]*sigma+rolling_noise(draws[2],self.patch)*sigma*self.block
        return Prepared(blurred+n,images,sigma,loss_weight=weight,model_kwargs={'augment_labels':augment_labels},metadata={'blur_sigma':blur},reduction='per_pixel' if self.native_loss else 'sum_leaves')

def relay_augmentation(spec):
    if spec is None:return None
    from .native_models import activate
    activate('relay',['training','torch_utils','dnnlib'])
    from training.augment import AugmentPipe
    return AugmentPipe(**spec)

@register('batch','edm')
class EDMBatch:
    def __init__(self,p_mean=-1.2,p_std=1.2,sigma_data=.5,augmentation=None,native_loss=False):
        self.mean=p_mean;self.std=p_std;self.data=sigma_data;self.augment=relay_augmentation(augmentation);self.native_loss=native_loss
    def __call__(self,images,t,stream,noise=None,context=None):
        if images.ndim==5 and self.augment is not None:raise ValueError('Video EDM requires an explicit temporally consistent augmentation adapter')
        shape=(len(images),*([1]*(images.ndim-1)))
        normal=stream.normal(torch.empty(*shape,device=images.device)) if t is None else t.reshape(shape)
        sigma=(normal*self.std+self.mean).exp();weight=(sigma**2+self.data**2)/(sigma*self.data)**2
        images,augment_labels=self.augment(images) if self.augment is not None else (images,None)
        noise=stream.normal(images) if noise is None else noise
        return Prepared(images+noise*sigma,images,sigma,loss_weight=weight,model_kwargs={'augment_labels':augment_labels},reduction='per_pixel' if self.native_loss else 'sum_leaves')

@register('batch','lapflow')
class LapFlowBatch:
    def __init__(self,levels=2,activations=None,up='nearest',noise_mode='native',curve='linear',epsilon=0.,weights=None,noise_var_factor=None,variance_aware=False):
        self.levels=levels;self.activations=activations or ([.4,0.] if levels==2 else [.6,.3,0.])
        if len(self.activations)!=levels:raise ValueError('Activation count mismatch')
        self.rep=build('representation',{'name':'laplacian','args':{'levels':levels,'up':up,'order':'fine_first'}})
        self.paths=[build('path',{'name':'delayed_flow','args':{'activation':a,'curve':curve,'epsilon':epsilon if i==levels-1 else 0.}}) for i,a in enumerate(self.activations)]
        self.mode=noise_mode;self.weights=weights or [1.]*levels
        self.noise_factor=noise_var_factor;self.variance=variance_aware
        if levels!=2 and (variance_aware or curve!='linear' or epsilon!=0. or noise_var_factor is not None):raise ValueError('Released three-level LapFlow supports only the linear, unscaled native noise path')
    def __call__(self,images,t,stream,noise=None,context=None):
        context=context or {};stage=int(context.get('stage',0));clean=self.rep.encode(images)
        if stage<0 or stage>=self.levels:raise ValueError('Invalid active LapFlow band')
        if torch.any(t<self.activations[stage]):raise ValueError('LapFlow path evaluated before band activation')
        if noise is None:
            full=stream.normal(images)
            if self.noise_factor is not None:full=full*self.noise_factor
            if self.levels==2 or self.mode=='decomposed':noise=self.rep.encode(full)
            else:noise=[full]+[stream.normal(c) for c in clean[1:]]
        samples=[p.sample(c,t,e) for p,c,e in zip(self.paths[stage:],clean[stage:],noise[stage:])]
        kwargs={'var_scale':None if stage==self.levels-1 else [e.std() for e in noise[stage:]]} if self.variance else {}
        weights=[1.] if self.levels==2 and stage==1 else self.weights[stage:]
        return Prepared(tuple(s.state for s in samples),[s.velocity for s in samples],t,model_kwargs=kwargs,leaf_weights=weights,metadata={'stage':stage,'representation':'laplacian_fine_first'},reduction='native_global')

@register('batch','dfm')
class DFMBatch:
    def __init__(self,levels=2,curve='linear',stage_weights=None,weighting='uniform',repa_weight=0.,native_debug_rng=False):
        self.levels=levels;self.curve=curve;self.stage_weights=stage_weights or ([.9,.1] if levels==2 else [1/levels]*levels);self.weighting=weighting
        self.repa_weight=repa_weight
        if not isinstance(native_debug_rng,bool):raise ValueError('Native DFM debug RNG policy must be boolean')
        self.native_debug_rng=native_debug_rng
        self.rep=build('representation',{'name':'laplacian','args':{'levels':levels,'up':'bilinear','order':'coarse_first'}})
        self.path=build('path',{'name':'flow','args':{'orientation':'data_to_noise','curve':curve}})
    def __call__(self,images,t,stream,noise=None,context=None):
        context=context or {};clean=self.rep.encode(images);batch=images.shape[0]
        stage=context.get('stage')
        if stage is None:
            u=stream.uniform((),device=images.device).item();stage=int(np.searchsorted(np.cumsum(self.stage_weights),u,side='right'))
        mask=torch.zeros(self.levels,device=images.device);mask[:stage+1]=1
        if t.ndim!=2:
            times=[]
            for i in range(self.levels):
                if i<stage:times.append(torch.sigmoid(stream.normal(torch.empty(batch,1,1,1,device=images.device))-2))
                elif i==stage:
                    if self.weighting=='uniform':times.append(t.reshape(-1,1,1,1))
                    else:
                        sig=stream.normal(torch.empty(batch,1,1,1,device=images.device)).exp()
                        times.append(sig/(1+sig) if self.curve=='linear' else 2/math.pi*torch.atan(sig))
                else:times.append(torch.ones(batch,1,1,1,device=images.device))
            t=torch.stack([u.flatten() for u in times],-1)
        noise=noise if noise is not None else [stream.normal(c) for c in clean]
        samples=[self.path.sample(c,t[:,i],e) for i,(c,e) in enumerate(zip(clean,noise))]
        prepared=REPAFeaturesPrepared if self.repa_weight else DFMPrepared
        extra={'native_debug_rng':self.native_debug_rng}
        if self.repa_weight:
            if 'teacher_features' not in context:raise ValueError('REPA requires explicitly supplied local teacher_features')
            features=context['teacher_features'];features=[features] if torch.is_tensor(features) else features
            extra.update(teacher_features=features,repa_weight=self.repa_weight)
        return prepared(dict(enumerate(s.state for s in samples)),dict(enumerate(s.velocity for s in samples)),t,model_kwargs={'drop_mask':mask},leaf_weights=mask.tolist(),metadata={'stage':stage},reduction='native_dims',**extra)

@register('batch','dimr')
class DiMRBatch:
    def __init__(self,steps=1000,beta_start=.00085,beta_end=.012,betas=None,levels=2,loss_weights=(.25,1.)):
        self.path=build('path',{'name':'discrete_vp','args':{'steps':steps,'beta_start':beta_start,'beta_end':beta_end,'betas':betas}})
        self.levels=levels;self.weights=list(loss_weights)
    def __call__(self,images,t,stream,noise=None,context=None):
        from .multiscale import dimr_targets
        noise=noise if noise is not None else stream.normal(images)
        # DiMR's training n is 1..N, whereas ADM indices are 0..N-1.
        sample=self.path.sample(images,t.long()-1,noise)
        targets=dimr_targets(noise,self.levels,True);keys=sorted(targets)
        return Prepared(sample.state,[targets[k] for k in keys],t.float()/1000*8-4,leaf_weights=self.weights,metadata={'native_time':'n/1000*8-4'})

@register('batch','mdm')
class MDMBatch:
    def __init__(self,gammas,scales=(2,1),schedule_shifted=True,power=1.,prediction='v',loss_target='epsilon',double_loss=True,weights=None,mixed_ratio=None):
        self.gammas=torch.tensor(gammas,dtype=torch.float32);self.scales=scales;self.shift=schedule_shifted;self.power=power;self.prediction=prediction;self.loss_target=loss_target;self.double=double_loss;self.weights=weights
        self.mixed=None
        if mixed_ratio is not None:
            values=np.asarray([float(x) for x in mixed_ratio.split(':')] if isinstance(mixed_ratio,str) else mixed_ratio)
            if len(values)!=len(scales) or np.any(values<=0):raise ValueError('Mixed ratios require one positive proportion per level')
            self.mixed=np.cumsum(values)/sum(values)
    def __call__(self,images,t,stream,noise=None,context=None):
        from .multiscale import snr_shift
        clean=[images]
        for previous,current in zip(self.scales[:-1],self.scales[1:]):clean.append(block_average(clean[-1],previous//current) if images.ndim==5 else torch.nn.functional.avg_pool2d(clean[-1],previous//current))
        noise=noise if noise is not None else [stream.normal(x) for x in clean]
        gamma=self.gammas.to(images.device)[t.long()].view(-1,*([1]*(images.ndim-1)))
        gs=[snr_shift(gamma,s,self.power) if self.shift else gamma for s in self.scales]
        clean=[x if self.shift else x/s for x,s in zip(clean,self.scales)]
        state=[g.sqrt()*x+(1-g).sqrt()*e for x,e,g in zip(clean,noise,gs)]
        targets=[e if self.loss_target=='epsilon' else g.sqrt()*e-(1-g).sqrt()*x for x,e,g in zip(clean,noise,gs)]
        weights=self.weights or [1. if i==0 or self.double else 0. for i in range(len(clean))]
        counts=None if self.mixed is None else [int(m*len(images)) for m in self.mixed]
        if counts is not None and min(counts)<1:raise ValueError('Mixed batch has an empty active level')
        masks=None if counts is None else [torch.arange(len(images),device=images.device)<n for n in counts]
        return Prepared(state,targets,t-1,leaf_weights=weights,metadata={'gammas':gs,'prediction':self.prediction,'loss_target':self.loss_target,'double_loss':self.double,'active_counts':counts},leaf_masks=masks,leaf_normalizers=self.mixed)

@register('batch','ppflow')
class PPFlowBatch:
    def __init__(self,levels=2,strict_native_sampling=True):
        if levels not in (2,3):raise ValueError('PPFlow release supports 2 or 3 stages')
        self.levels=levels;self.strict=strict_native_sampling
    def __call__(self,images,t,stream,noise=None,context=None):
        from .operators import channel_down
        b=len(images);device=images.device
        if b%(2 if self.levels==2 else 4):raise ValueError('Native PPFlow balanced batches must divide stage counts exactly')
        durations=torch.tensor([0.,.5,1.] if self.levels==2 else [0.,.5,.75,1.],device=device,dtype=images.dtype)
        ratios=torch.diff(durations,prepend=durations[:1])
        noise=stream.normal(images) if noise is None else noise
        if t is None:
            if self.strict:
                candidates=stream.uniform((20*b,),device=device)
                parts=[candidates[(candidates>=a)&(candidates<=z)][:round((b*ratio).item())] for a,z,ratio in zip(durations[:-1],durations[1:],ratios[1:])]
                if sum(len(part) for part in parts)!=b:raise ValueError('Original PPFlow rejection sampler produced too few times; use a larger batch or explicit stratified controlled-transplant sampler')
                t=torch.cat(parts)
            else:
                t=torch.cat([a+(z-a)*stream.uniform((round((b*ratio).item()),),device=device) for a,z,ratio in zip(durations[:-1],durations[1:],ratios[1:])])
        if self.levels==2:clean=[channel_down(images),images];eps=[channel_down(noise),noise]
        else:clean=[channel_down(images),channel_down(images,width_only=True),images];eps=[channel_down(noise),channel_down(noise,width_only=True),noise]
        states=[];targets=[]
        for i,(x,e,a,z) in enumerate(zip(clean,eps,durations[:-1],durations[1:])):
            start=round((durations[i]*b).item());end=round((durations[i+1]*b).item())
            start_state=(1.-a)*e+a*x;end_state=(1.-z)*e+z*x
            u=t[start:end,None,None,None]
            state=(u-a)/(z-a)*end_state[start:end]+(z-u)/(z-a)*start_state[start:end]
            velocity=(end_state[start:end]-start_state[start:end])/(z-a)
            states.extend(state.unbind());targets.extend(velocity.unbind())
        return Prepared(states,targets,t,model_kwargs={'x_ratio':ratios,'x_duration':durations,'training':True},reduction='examples')

@register('batch','pixelflow')
class PixelFlowBatch:
    def __init__(self,stages=4,train_steps=1000,patch=2,attention_head_dim=16,gamma=-1/3):
        from .clocks import PixelFlowClock
        self.clock=PixelFlowClock(train_steps,stages,gamma);self.stages=stages;self.patch=patch;self.head_dim=attention_head_dim
    def __call__(self,images,t,stream,noise=None,context=None):
        import random
        from einops import rearrange
        from .position import rotary_2d as get_2d_rotary_pos_embed
        from .paths import EndpointFlow
        from .operators import resize
        context=context or {};batch=len(images);indices=context.get('stage_indices')
        if indices is None:
            indices=(list(range(self.stages))*(batch//self.stages+1))[:batch];random.shuffle(indices)
            indices=torch.tensor(indices,device=images.device)
        t=t.long();states=[];targets=[];times=[];lengths=[];ropes=[];order=[]
        for down_index in range(self.stages):
            stage=self.stages-down_index-1;selected=indices==stage
            ids=torch.where(selected)[0];clean=images[selected]
            for _ in range(down_index):clean=resize(clean,scale=.5,mode='bilinear')
            eps=stream.normal(clean) if noise is None else noise[down_index]
            local=t[selected].double()/self.clock.train_steps
            path=EndpointFlow(self.clock.starts[stage],self.clock.ends[stage],projection={'down':'bilinear','up':'nearest'})
            sample=path.sample(clean,local.float(),eps)
            states.append(rearrange(sample.state,'b c (h ph) (w pw) -> (b h w) (c ph pw)',ph=self.patch,pw=self.patch))
            targets.append(rearrange(sample.velocity,'b c (h ph) (w pw) -> (b h w) (c ph pw)',ph=self.patch,pw=self.patch))
            times.append(self.clock.network_times[stage][t[selected].cpu()].float().to(images.device))
            grid=clean.shape[-1]//self.patch;length=grid*grid
            rope=get_2d_rotary_pos_embed(embed_dim=self.head_dim,crops_coords=((0,0),(grid,grid)),grid_size=(grid,grid))
            lengths.extend([length]*len(clean));ropes.extend([torch.stack(rope,-1)]*len(clean));order.extend(ids.tolist())
        cumulative=torch.cumsum(torch.tensor([0]+lengths,device=images.device),0).int()
        latent_size=torch.tensor([int(math.sqrt(n)) for n in lengths],device=images.device,dtype=torch.int32)
        pos=torch.cat(ropes).float().to(images.device)
        model_kwargs={'latent_size':latent_size,'pos_embed':pos,'cu_seqlens_q':cumulative,'cu_seqlens_k':None,'seqlen_list_q':lengths,'seqlen_list_k':None}
        labels=context.get('class_labels')
        if labels is not None:model_kwargs['class_labels']=labels[order]
        return Prepared(torch.cat(states),torch.cat(targets),torch.cat(times),model_kwargs,segment_lengths=lengths,metadata={'order':order,'stage_indices':indices},reduction='native_packed')

@register('batch','ssd')
class SSDBatch:
    def __init__(self,resolutions,betas=None,unet_res_layer_map=None,loss_weight='constant',rescale_timesteps=False):
        from .operators import BilinearOperator
        self.resolutions=list(resolutions);self.map={int(k):v for k,v in (unet_res_layer_map or {r:r for r in resolutions}).items()}
        self.path=build('path',{'name':'discrete_vp','args':{'steps':len(resolutions),'betas':betas}})
        self.ops=[BilinearOperator((resolutions[max(i-1,0)],)*2,(r,)*2) for i,r in enumerate(resolutions)]
        self.weight=loss_weight;self.rescale=rescale_timesteps
    def cumulative(self,x,index):
        for op in self.ops[:index+1]:x=op(x)
        return x
    def model_input(self,state,index):
        from .operators import resize
        previous=max(index-1,0);rt=self.resolutions[index];rs=self.resolutions[previous]
        cond=None if rt==rs else torch.full((len(state),),rs,device=state.device,dtype=torch.float32)
        if cond is not None and rs==max(self.map.values()):state=resize(state,size=(rs,rs),mode='bilinear',antialias=True);cond=None
        return state,cond,rs,rt
    def __call__(self,images,t,stream,noise=None,context=None):
        if not torch.all(t==t[0]):
            pairs={(self.resolutions[max(int(i)-1,0)],self.resolutions[int(i)]) for i in t}
            if len(pairs)!=1 or next(iter(pairs))[0]!=next(iter(pairs))[1]:raise ValueError('SSD multi-time batches must share a constant-resolution path; transitions require one time')
        index=int(t[0]);previous=max(index-1,0)
        clean_t=self.cumulative(images,index);clean_s=self.cumulative(images,previous) if previous else images
        eps=stream.normal(clean_t) if noise is None else noise
        sample=self.path.sample(clean_t,t,eps);state,cond,rs,rt=self.model_input(sample.state,index)
        weight=1.
        if self.weight!='constant':
            snr=(sample.alpha/sample.sigma).flatten()**2
            if self.weight=='snr':weight=snr
            elif self.weight=='inv_snr':weight=1/snr
            elif self.weight=='trunc_snr':weight=snr.clamp_min(1)
            elif self.weight.startswith('min_snr_'):weight=snr.clamp_max(float(self.weight.removeprefix('min_snr_')))
            elif self.weight.startswith('max_snr_'):weight=snr.clamp_min(float(self.weight.removeprefix('max_snr_')))
            else:raise ValueError('Unsupported SSD loss weight')
            weight[snr==0]=1.
        return Prepared(state,clean_s,t.float()*1000/len(self.path.betas) if self.rescale else t,model_kwargs={'cond_rs':cond},loss_weight=weight,metadata={'rs':rs,'rt':rt,'expected_output_size':self.map[rs]})
