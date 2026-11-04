"""Conditional paths. Time orientation is explicit, never inferred from a name."""
from dataclasses import dataclass,field
import math
import numpy as np
import torch
from .tree import expand
from .registry import register
from .operators import Projection,heat_blur

@dataclass
class PathSample:
    state:torch.Tensor
    clean:torch.Tensor
    noise:torch.Tensor
    effective_clean:torch.Tensor
    alpha:torch.Tensor
    sigma:torch.Tensor
    velocity:torch.Tensor|None=None
    low:torch.Tensor|None=None
    high:torch.Tensor|None=None
    dlow:torch.Tensor|None=None
    dhigh:torch.Tensor|None=None
    dsigma:torch.Tensor|None=None
    drift:torch.Tensor|None=None
    meta:dict=field(default_factory=dict)

class BandPath:
    def __init__(self,projection=None):self.projection=Projection(**(projection or {}))
    def coefficients(self,t):raise NotImplementedError
    def sample(self,x,t,noise):
        if x.shape!=noise.shape:raise ValueError('Path noise shape mismatch')
        t=expand(t,x)
        low,high,sigma,dlow,dhigh,dsigma=self.coefficients(t)
        px=self.projection(x);hx=x-px
        mean=low*px+high*hx
        velocity=dlow*px+dhigh*hx+dsigma*noise
        # Effective clean refers to the alpha=low signal normalization.
        # Do not clamp small signal coefficients: that changes the target. At
        # a common zero, use the differentiable path limit when identifiable.
        safe_low=torch.where(low!=0,low,torch.ones_like(low))
        safe_dlow=torch.where(dlow!=0,dlow,torch.ones_like(dlow))
        limit=(dlow*px+dhigh*hx)/safe_dlow
        undefined=(low==0)&((high!=0)|(dlow==0))
        effective=torch.where(low!=0,mean/safe_low,limit)
        effective=torch.where(undefined,torch.full_like(effective,float('nan')),effective)
        return PathSample(mean+sigma*noise,x,noise,effective,low,sigma,velocity,
                          low,high,dlow,dhigh,dsigma,px-x)
    def clean_from_velocity(self,z,v,t):
        if not self.projection.orthogonal:
            raise ValueError('Analytic band inverse requires an orthogonal projection; configure an explicit decoder for this operator')
        low,high,sigma,dlow,dhigh,dsigma=self.coefficients(expand(t,z))
        a=dsigma*low-sigma*dlow;b=dsigma*high-sigma*dhigh
        if torch.any(a.abs()<1e-12) or torch.any(b.abs()<1e-12):
            raise ValueError('Clean data is not identifiable from state and velocity at this time')
        y=dsigma*z-sigma*v
        py=self.projection(y)
        return py/a+(y-py)/b

@register('path','flow')
class Flow(BandPath):
    def __init__(self,orientation='noise_to_data',curve='linear',projection=None):
        super().__init__(projection)
        if orientation not in ('noise_to_data','data_to_noise'):raise ValueError('Invalid orientation')
        if curve not in ('linear','cosine'):raise ValueError('Invalid flow curve')
        self.orientation=orientation;self.curve=curve
    def coefficients(self,t):
        if self.orientation=='data_to_noise':
            if self.curve=='linear':
                a=1-t;s=t;da=-torch.ones_like(t);ds=torch.ones_like(t)
            else:
                a=torch.cos(t*math.pi/2);s=torch.sin(t*math.pi/2)
                da=-math.pi/2*torch.sin(t*math.pi/2);ds=math.pi/2*torch.cos(t*math.pi/2)
            return a,a,s,da,da,ds
        u=t if self.orientation=='noise_to_data' else 1-t
        direction=1 if self.orientation=='noise_to_data' else -1
        if self.curve=='linear':a=u;s=1-u;da=torch.ones_like(t)*direction;ds=-da
        else:
            a=torch.sin(u*math.pi/2);s=torch.cos(u*math.pi/2)
            da=direction*math.pi/2*s;ds=-direction*math.pi/2*a
        return a,a,s,da,da,ds
    def sample(self,x,t,noise):
        a,_,s,da,_,ds=self.coefficients(expand(t,x))
        return PathSample(a*x+s*noise,x,noise,x,a,s,da*x+ds*noise,a,a,da,da,ds)
    def clean_from_velocity(self,z,v,t):
        a,_,s,da,_,ds=self.coefficients(expand(t,z))
        return (ds*z-s*v)/(ds*a-s*da)

@register('path','rdd_flow')
class RDDFlow(BandPath):
    def __init__(self,start=0.,end=1.,projection=None,drift=True):
        super().__init__(projection);self.start=start;self.end=end;self.drift=drift
        if not 0<=start<end<=1:raise ValueError('Invalid RDD flow stage')
    def coefficients(self,t):
        beta=t*(self.end-t)/(self.end-self.start) if self.drift else torch.zeros_like(t)
        dbeta=(self.end-2*t)/(self.end-self.start) if self.drift else torch.zeros_like(t)
        return t,t-beta,1-t,torch.ones_like(t),1-dbeta,-torch.ones_like(t)
    def sample(self,x,t,noise):
        # Preserve native operation order for fp32 parity.
        out=super().sample(x,t,noise);u=expand(t,x)
        beta=u*((self.end-u)/(self.end-self.start)) if self.drift else 0
        dbeta=(self.end-2*u)/(self.end-self.start) if self.drift else 0
        d=self.projection(x)-x
        out.state=u*x+(1-u)*noise+beta*d
        out.velocity=x-noise+dbeta*d
        return out
    def clean_from_velocity(self,z,v,t):
        u=expand(t,z).to(dtype=z.dtype)
        y=z+(1.0-u)*v
        if not self.drift:return y
        if not self.projection.orthogonal:
            raise ValueError('RDD-FM analytic clean inverse requires an orthogonal projection; configure an explicit decoder for this operator')
        beta=u*(self.end-u)/(self.end-self.start)
        derivative=(self.end-2.0*u)/(self.end-self.start)
        denominator=1.0-(beta+(1.0-u)*derivative)
        if torch.any(denominator.abs()<1e-6):raise ValueError('RDD-FM high band is not identifiable at the requested time')
        low=self.projection(y)
        return low+(y-low)/denominator

@register('path','stage_flow')
class StageFlow(Flow):
    """Linear flow evaluated through finite stage endpoints, as in PPFlow.

    It has the same real-valued path as Flow but preserves the released
    interpolation and derivative operation order in finite precision.
    """
    def __init__(self,start=0.,end=1.):
        super().__init__();self.start=start;self.end=end
        if not 0<=start<end<=1:raise ValueError('Invalid stage endpoints')
    def sample(self,x,t,noise):
        u=expand(t,x);a=self.start;b=self.end
        start=a*x+(1-a)*noise;end=b*x+(1-b)*noise
        state=(u-a)/(b-a)*end+(b-u)/(b-a)*start
        velocity=(1*end+-1*start)/(b-a)
        return PathSample(state,x,noise,x,u,1-u,velocity,u,u,torch.ones_like(u),torch.ones_like(u),-torch.ones_like(u))

@register('path','endpoint_flow')
class EndpointFlow(BandPath):
    """PixelFlow noisy endpoints, with velocity in local tau or global time."""
    def __init__(self,start=0.,end=1.,projection=None,velocity_clock='local'):
        super().__init__(projection);self.start=start;self.end=end;self.velocity_clock=velocity_clock
        if end<=start or velocity_clock not in ('local','global'):raise ValueError('Invalid endpoints')
    def coefficients(self,t):
        tau=t if self.velocity_clock=='local' else (t-self.start)/(self.end-self.start)
        factor=1 if self.velocity_clock=='local' else 1/(self.end-self.start)
        lo=(1-tau)*self.start+tau*self.end;hi=tau*self.end
        s=(1-tau)*(1-self.start)+tau*(1-self.end)
        return lo,hi,s,torch.ones_like(t)*(self.end-self.start)*factor,torch.ones_like(t)*self.end*factor,torch.ones_like(t)*(self.start-self.end)*factor
    def sample(self,x,t,noise):
        out=super().sample(x,t,noise)
        u=expand(t,x);tau=u if self.velocity_clock=='local' else (u-self.start)/(self.end-self.start)
        start=self.start*self.projection(x)+(1-self.start)*noise
        end=self.end*x+(1-self.end)*noise
        out.state=tau*end+(1-tau)*start
        out.velocity=(end-start)/(1 if self.velocity_clock=='local' else self.end-self.start)
        return out

@register('path','delayed_flow')
class DelayedFlow(Flow):
    """LapFlow global clock: data activates at a, noise retains global sigma."""
    def __init__(self,activation=0.,curve='linear',epsilon=0.):
        super().__init__(curve='cosine' if curve=='square' else curve);self.curve=curve;self.activation=activation;self.epsilon=epsilon
        if not 0<=activation<1:raise ValueError('Invalid activation')
    def coefficients(self,t):
        a=(t-self.activation)/(1-self.activation)
        if self.curve=='linear':
            s=1-(1-self.epsilon)*t;da=torch.ones_like(t)/(1-self.activation);ds=torch.ones_like(t)*(self.epsilon-1)
        elif self.curve=='square':
            a=torch.cos((1-t)**2/(1-self.activation)*math.pi/2)
            s=torch.sin((1-t)**2*math.pi/2)
            if self.activation==0:
                da=(1-t)*math.pi*torch.sin((1-t)**2*math.pi/2);ds=-(1-t)*math.pi*torch.cos((1-t)**2*math.pi/2)
            else:
                factor=math.pi/(2*(1-self.activation))
                da=torch.sin(((1-t)**2)*factor)*(2*(1-t)*factor)
                factor=math.pi/2;ds=-torch.cos(((1-t)**2)*factor)*(2*(1-t)*factor)
        else:
            a=torch.cos((1-t)/(1-self.activation)*math.pi/2)
            da=math.pi/2/(1-self.activation)*torch.sin((1-t)/(1-self.activation)*math.pi/2)
            s=torch.sin((1-t)*math.pi/2);ds=-math.pi/2*torch.cos((1-t)*math.pi/2)
        return a,a,s,da,da,ds

@register('path','vp')
class VP(Flow):
    """Continuous VP and explicit SNR rescaling (DCTdiff)."""
    def __init__(self,beta_min=.1,beta_max=20.,snr_scale=1.):
        self.beta_min=beta_min;self.beta_max=beta_max;self.snr_scale=snr_scale
        if snr_scale<=0:raise ValueError('SNR scale must be positive')
    def coefficients(self,t):
        integral=self.beta_min*t+(self.beta_max-self.beta_min)*(t**2)*.5
        raw=torch.exp(-1*(self.beta_min*t+.5*(self.beta_max-self.beta_min)*(t**2)))
        # Same expression and ordering as author's adjusted integral.
        integral=integral+(torch.log(1+(self.snr_scale-1)*raw)-math.log(self.snr_scale))
        abar=torch.exp(-integral);a=abar**.5;s=(1-abar)**.5
        beta=(self.beta_min+(self.beta_max-self.beta_min)*t)/(1+(self.snr_scale-1)*raw)
        da=-beta*a/2;ds=beta*a*a/(2*s.clamp_min(1e-12))
        return a,a,s,da,da,ds

@register('path','discrete_vp')
class DiscreteVP:
    def __init__(self,steps=1000,beta_start=.0001,beta_end=.02,betas=None,
                 drift_start=None,drift_end=None,drift_multiplier=1.,projection=None,timestep_map=None):
        self.betas=np.asarray(betas if betas is not None else np.linspace(beta_start,beta_end,steps),dtype=np.float64)
        if self.betas.ndim!=1 or np.any(self.betas<=0) or np.any(self.betas>1):raise ValueError('Invalid betas')
        self.abar=np.cumprod(1-self.betas)
        self.alpha=np.sqrt(self.abar);self.sigma=np.sqrt(1-self.abar)
        self.drift_start=drift_start;self.drift_end=drift_end;self.multiplier=drift_multiplier
        if (drift_start is None)!=(drift_end is None) or (drift_start is not None and drift_end<=drift_start):raise ValueError('Invalid drift stage')
        self.projection=Projection(**(projection or {}))
        self.timestep_map=np.arange(len(self.betas)) if timestep_map is None else np.array(timestep_map)
    def extract(self,a,t,x):
        # ADM/RDD source intentionally casts coefficient tables to fp32.
        return torch.from_numpy(a).to(device=x.device)[t.long()].float().reshape(-1,*([1]*(x.ndim-1)))
    def sample(self,x,t,noise):
        a=self.extract(self.alpha,t,x);s=self.extract(self.sigma,t,x)
        if self.drift_start is None:drift=torch.zeros_like(x);lam=0.
        else:
            drift=self.projection(x)-x
            mapped=torch.as_tensor(self.timestep_map,device=t.device,dtype=t.dtype)[t.long()]
            lam=expand((mapped-self.drift_start)/(self.drift_end-self.drift_start),x)*self.multiplier
        effective=x+lam*drift
        if self.drift_start is None:state=a*x+s*noise
        else:
            mapped=torch.as_tensor(self.timestep_map,device=t.device,dtype=t.dtype)[t.long()]
            mapped=mapped.reshape(-1,*([1]*(x.ndim-1)))
            state=a*x+self.multiplier*a*(mapped-self.drift_start)/(self.drift_end-self.drift_start)*drift+s*noise
        # Native RDD builds the training noise with sqrt(abar/(1-abar)),
        # not sqrt(abar)/sqrt(1-abar), and avoids cancelling (effective-x).
        # Keep that floating operation tree as the path's target coordinate.
        if self.drift_start is None:effective_noise=noise
        else:
            factor=expand((mapped.reshape(-1)-self.drift_start)/(self.drift_end-self.drift_start),x)
            abar=self.extract(self.abar,t,x)
            effective_noise=noise+self.multiplier*torch.sqrt(abar/(1-abar))*factor*drift
        return PathSample(state,x,noise,effective,a,s,drift=drift,meta={'lambda':lam,'index':t,'effective_noise':effective_noise})

@register('path','heat_edm')
class HeatEDM:
    def __init__(self,patch=4,min_scale=.001):self.patch=patch;self.min_scale=min_scale
    def sample(self,x,sigma,noise,blur_sigma=None):
        if blur_sigma is None:raise ValueError('Relay path needs jointly sampled blur and noise sigma')
        s=expand(sigma,x);blur=expand(blur_sigma,x)
        f=heat_blur(x,blur,self.patch,self.min_scale)
        return PathSample(f+s*noise,x,noise,f,torch.ones_like(s),s,meta={'blur_sigma':blur})
