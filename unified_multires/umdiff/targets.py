"""Prediction semantics are independent of path, heads and loss coordinates."""
import torch
from .registry import register
from .tree import mean_flat

@register('target','joint')
class Joint:
    def __init__(self,quantity='velocity'):
        if quantity not in ('velocity','epsilon','effective_epsilon','clean','effective_clean','v'):raise ValueError('Unknown target')
        self.quantity=quantity
    def encode(self,s):
        if self.quantity in ('effective_epsilon','effective_clean','v') and not torch.isfinite(s.effective_clean).all():
            raise ValueError('Effective clean target is undefined at this path endpoint')
        if self.quantity=='velocity':
            if s.velocity is None:raise ValueError('Path has no continuous velocity')
            return s.velocity
        if self.quantity=='epsilon':return s.noise
        if self.quantity=='effective_epsilon':
            if s.meta is not None and 'effective_noise' in s.meta:return s.meta['effective_noise']
            return s.noise+s.alpha/s.sigma*(s.effective_clean-s.clean)
        if self.quantity=='clean':return s.clean
        if self.quantity=='effective_clean':return s.effective_clean
        return s.alpha*s.noise-s.sigma*s.effective_clean
    def clean(self,pred,state,t,path):
        if self.quantity=='clean':return pred
        if self.quantity=='velocity':return path.clean_from_velocity(state,pred,t)
        # Other decoders need explicit path coefficients; no heuristic clamping.
        if hasattr(path,'coefficients'):
            from .tree import expand
            a,hi,s,*_=path.coefficients(expand(t,state))
        elif hasattr(path,'extract'):
            a=path.extract(path.alpha,t,state);s=path.extract(path.sigma,t,state)
        else:raise ValueError('This target needs an explicit decoder')
        if self.quantity=='effective_epsilon':return (state-s*pred)/a
        if self.quantity=='v':
            effective=(a*state-s*pred)/(a*a+s*s)
        elif self.quantity=='epsilon':effective=(state-s*pred)/a
        elif self.quantity=='effective_clean':effective=pred
        else:raise ValueError('Unsupported decoder')
        if hasattr(path,'coefficients'):
            if torch.equal(hi,a):return effective
            if not path.projection.orthogonal:raise ValueError('Transformed-clean inverse needs an orthogonal projection')
            if torch.any(hi.abs()<1e-12):raise ValueError('High-frequency clean is unidentifiable at this time')
            low=path.projection(effective)
            return low+(effective-low)*a/hi
        if path.drift_start is None:return effective
        mapped=torch.as_tensor(path.timestep_map,device=t.device,dtype=t.dtype)[t.long()]
        lam=(mapped-path.drift_start)/(path.drift_end-path.drift_start)*path.multiplier
        from .tree import expand
        if torch.any((1-lam).abs()<1e-12):raise ValueError('Clean inverse is singular at stage start')
        low=path.projection(effective)
        return low+(effective-low)/(1-expand(lam,state))

@register('target','decoupled_clean')
class DecoupledClean:
    """f-DM-style separate transformed signal and removed detail predictions."""
    def encode(self,s):return {'signal':s.effective_clean,'residual':s.clean-s.effective_clean}
    def clean(self,pred,state,t,path):return pred['signal']+pred['residual']

@register('target','rdd_decoupled')
class RDDDecoupled:
    """Current RDD extra drift head, with original effective-epsilon main head."""
    def encode(self,s):return {'main':Joint('effective_epsilon').encode(s),'drift':s.drift}
    def clean(self,pred,state,t,path):return Joint('effective_epsilon').clean(pred['main'],state,t,path)

@register('metric','mse')
class MSE:
    def __init__(self,head_weights=None,element_weights=None,dct_statistics=None,reduction='flatten'):
        self.head_weights=head_weights or {};self.element_weights=element_weights
        if reduction not in ('flatten','native_dims'):raise ValueError('Unknown MSE reduction arithmetic')
        self.reduction=reduction
        if dct_statistics is not None:
            if element_weights is not None:raise ValueError('Choose explicit weights or DCT statistics')
            from .dctcodec import frequency_weights
            self.element_weights=frequency_weights(**dct_statistics)
    def __call__(self,pred,target,weight=1.):
        if isinstance(target,dict):
            if not isinstance(pred,dict) or set(pred)!=set(target):raise ValueError('Prediction head names differ from target')
            terms={k:self._one(pred[k],target[k])*self.head_weights.get(k,1.) for k in target}
            return sum(terms.values())*weight,terms
        loss=self._one(pred,target)
        return loss*weight,{'mse':loss}
    def _one(self,a,b):
        if a.shape!=b.shape:raise ValueError(f'Loss shape mismatch {a.shape} vs {b.shape}')
        error=(a-b)**2
        if self.element_weights is not None:error=error*torch.as_tensor(self.element_weights,device=a.device,dtype=a.dtype)
        return error.mean(dim=list(range(1,error.ndim))) if self.reduction=='native_dims' else mean_flat(error)

@register('precondition','edm')
class EDMPrecondition:
    def __init__(self,sigma_data=.5):self.sigma_data=sigma_data
    def __call__(self,net,x,sigma,**kwargs):
        x=x.float();sigma=sigma.float()
        s=sigma.reshape(-1,*([1]*(x.ndim-1)));d=self.sigma_data
        skip=d*d/(s*s+d*d);out=s*d/(s*s+d*d).sqrt();cin=1/(s*s+d*d).sqrt()
        raw=net(cin*x,(sigma.log()/4).flatten(),**kwargs)
        return skip*x+out*raw
