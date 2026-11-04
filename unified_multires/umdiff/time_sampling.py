"""Training-time sampling is separate from path and network clocks."""
import random
import numpy as np
import torch
from .registry import register

@register('time_sampler','uniform')
class Uniform:
    def __init__(self,start=0.,end=1.):
        if not start<end:raise ValueError('Uniform interval is empty')
        self.start=start;self.end=end
    def __call__(self,batch,device,stream):return self.start+(self.end-self.start)*stream.uniform((batch,),device=device),{}

@register('time_sampler','discrete')
class Discrete:
    def __init__(self,steps=1000,offset=0,one_for_batch=False,rng='torch',vdm_gammas=None):
        self.steps=steps;self.offset=offset;self.one=one_for_batch;self.rng=rng
        if rng not in ('torch','numpy'):raise ValueError('Unknown discrete sampling RNG')
        self.weights=None
        if vdm_gammas is not None:
            gamma=torch.tensor(vdm_gammas,dtype=torch.float32)
            if offset!=1 or len(gamma)!=steps+1:raise ValueError('Native VDM weights require one-based time and steps+1 gammas')
            g=gamma[2:];previous=gamma[1:-1];weights=previous*(1-g)/(1-previous)/g-1
            self.weights=torch.cat([weights[:1],weights[:1],weights])
    def __call__(self,batch,device,stream):
        shape=(1 if self.one else batch,)
        value=torch.from_numpy(np.random.choice(list(range(self.offset,self.offset+self.steps)),shape)).to(device) if self.rng=='numpy' else stream.randint(self.offset,self.offset+self.steps,shape,device=device)
        value=value.expand(batch)
        return value,{} if self.weights is None else {'loss_weights':self.weights.to(device)[value]}

@register('time_sampler','lapflow')
class LapFlowTimes:
    def __init__(self,activations=(.4,0.),stage=None):
        if stage is not None and (type(stage) is not int or not 0<=stage<len(activations)):
            raise ValueError('Invalid fixed LapFlow training stage')
        self.activations=activations;self.stage=stage
    def __call__(self,batch,device,stream):
        # Released MoT chooses one suffix uniformly for the whole batch.
        t=stream.uniform((batch,),device=device)
        stage=random.randint(0,len(self.activations)-1) if self.stage is None else self.stage
        a=self.activations[stage]
        return a+t*(1-a),{'stage':stage}

@register('time_sampler','discrete_subset')
class DiscreteSubset:
    """Uniform conditional time within an explicitly listed resolution phase."""
    def __init__(self,indices,steps=1000,one_for_batch=False):
        if not indices or any(type(i) is not int or not 0<=i<steps for i in indices) or len(set(indices))!=len(indices):
            raise ValueError('Discrete subset needs unique valid training indices')
        self.indices=tuple(indices);self.one=one_for_batch
    def __call__(self,batch,device,stream):
        index=stream.randint(0,len(self.indices),(1 if self.one else batch,),device=device)
        return torch.tensor(self.indices,device=device,dtype=torch.long)[index].expand(batch),{}

@register('time_sampler','none')
class NoneTime:
    def __call__(self,batch,device,stream):return None,{}

@register('time_sampler','loss_second_moment')
class LossSecondMoment:
    """Independent ADM/RDD importance clock, including persistent loss history."""
    def __init__(self,steps=1000,history_per_term=10,uniform_prob=.001,adaptive=True):
        if steps<1 or history_per_term<1 or not 0<=uniform_prob<=1:raise ValueError('Invalid loss-aware clock')
        self.steps=steps;self.history_per_term=history_per_term;self.uniform_prob=uniform_prob;self.adaptive=adaptive
        self.history=np.zeros((steps,history_per_term),dtype=np.float64);self.counts=np.zeros(steps,dtype=np.int64)
    @property
    def requires_checkpoint_state(self):return self.adaptive
    def weights(self):
        if not self.adaptive or not (self.counts==self.history_per_term).all():return np.ones(self.steps,dtype=np.float64)
        weights=np.sqrt(np.mean(self.history**2,axis=-1));weights/=np.sum(weights);weights*=1-self.uniform_prob;weights+=self.uniform_prob/len(weights)
        return weights
    def __call__(self,batch,device,stream):
        probabilities=self.weights();probabilities=probabilities/np.sum(probabilities)
        indices=np.random.choice(len(probabilities),size=batch,p=probabilities)
        correction=1/(len(probabilities)*probabilities[indices])
        return torch.from_numpy(indices).long().to(device),{'importance_weights':torch.from_numpy(correction).float().to(device),**({'loss_observer':self.update} if self.adaptive else {})}
    def update(self,times,losses):
        if torch.distributed.is_initialized():
            gathered=[None]*torch.distributed.get_world_size();torch.distributed.all_gather_object(gathered,(times.tolist(),losses.tolist()))
        else:gathered=[(times.tolist(),losses.tolist())]
        for ts,ls in gathered:
            for t,loss in zip(ts,ls):
                if self.counts[t]==self.history_per_term:self.history[t,:-1]=self.history[t,1:];self.history[t,-1]=loss
                else:self.history[t,self.counts[t]]=loss;self.counts[t]+=1
    def state_dict(self):return dict(format=1,steps=self.steps,history_per_term=self.history_per_term,uniform_prob=self.uniform_prob,adaptive=self.adaptive,history=self.history.copy(),counts=self.counts.copy())
    def load_state_dict(self,state):
        for k in ('format','steps','history_per_term','uniform_prob','adaptive'):
            expected=1 if k=='format' else getattr(self,k)
            if state[k]!=expected:raise ValueError('Loss-aware clock contract mismatch')
        history=np.asarray(state['history']);counts=np.asarray(state['counts'])
        if history.shape!=self.history.shape or counts.shape!=self.counts.shape or history.dtype!=self.history.dtype or counts.dtype!=self.counts.dtype:
            raise ValueError('Loss-aware clock history shape/dtype mismatch')
        if not np.isfinite(history).all() or (counts<0).any() or (counts>self.history_per_term).any():
            raise ValueError('Invalid loss-aware clock history/counts')
        self.history[:]=history;self.counts[:]=counts

@register('time_sampler','dfm')
class DFMTimes:
    def __init__(self,weights=(.9,.1),weighting='uniform',curve='linear',stage=None):
        if weighting not in ('uniform','lognormal') or curve not in ('linear','cosine'):raise ValueError('Unsupported native DFM time policy')
        if stage is not None and (type(stage) is not int or not 0<=stage<len(weights)):
            raise ValueError('Invalid fixed DFM training stage')
        self.weights=weights;self.weighting=weighting;self.curve=curve;self.stage=stage
    def __call__(self,batch,device,stream):
        stage=int(np.random.choice(len(self.weights),p=self.weights)) if self.stage is None else self.stage
        times=[]
        for i in range(len(self.weights)):
            ref=torch.empty(batch,1,1,1,device=device)
            if i<stage:time=torch.sigmoid(stream.normal(ref)-2)
            elif i==stage:
                if self.weighting=='uniform':time=stream.uniform(ref.shape,device=device)
                else:
                    sigma=stream.normal(ref).exp()
                    time=sigma/(1+sigma) if self.curve=='linear' else 2/np.pi*torch.atan(sigma)
            else:time=torch.ones_like(ref)
            times.append(time.flatten())
        return torch.stack(times,-1),{'stage':stage}

@register('time_sampler','ssd')
class SSDTimes:
    """Author resolution-group sampling, including its importance weights."""
    def __init__(self,resolutions,weights=None,native_numpy=True):
        self.resolutions=np.asarray(resolutions);self.p=np.asarray(weights if weights is not None else np.ones(len(resolutions)),dtype=float)
        if len(self.p)!=len(resolutions) or (self.p<=0).any():raise ValueError('Invalid SSD sampling weights')
        self.p=self.p/self.p.sum();self.native_numpy=native_numpy
    def __call__(self,batch,device,stream):
        def choose(values,probabilities,size):
            if self.native_numpy:return np.random.choice(values,size=size,p=probabilities)
            cdf=torch.as_tensor(np.cumsum(probabilities),device=device)
            positions=torch.searchsorted(cdf,stream.uniform((size,),device=device).double()).cpu().numpy()
            return np.asarray(values)[positions]
        chosen=int(choose(np.arange(len(self.p)),self.p,1)[0]);previous=max(chosen-1,0)
        if self.resolutions[chosen]!=self.resolutions[previous]:indices=np.full(batch,chosen)
        else:
            all_t=np.arange(len(self.p));all_rs=self.resolutions[np.maximum(all_t-1,0)]
            valid=all_t[(self.resolutions==self.resolutions[chosen])&(all_rs==self.resolutions[chosen])]
            probability=self.p[valid];probability=probability/probability.sum();indices=choose(valid,probability,batch)
        weights=torch.from_numpy(1/(len(self.p)*self.p[indices])).float().to(device)
        return torch.from_numpy(indices).long().to(device),{'importance_weights':weights}

@register('time_sampler','logit_normal')
class LogitNormal:
    def __init__(self,mean=0.,std=1.,start=0.,end=1.):
        if std<=0 or start>=end:raise ValueError('Invalid logit-normal time settings')
        self.mean=mean;self.std=std;self.start=start;self.end=end
    def __call__(self,batch,device,stream):
        t=torch.sigmoid(stream.normal(torch.empty(batch,device=device))*self.std+self.mean)
        return self.start+(self.end-self.start)*t,{}
