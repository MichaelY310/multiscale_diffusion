"""Scale-space Gaussian posterior with matrix-free covariance square root."""
import torch
from .operators import BilinearOperator
from .registry import register

def lanczos_sqrt(action,vector,iterations=40,reorthogonalize=False):
    shape=vector.shape;b=vector.reshape(-1);norm=b.norm().clamp_min(1e-32)
    q=b/norm;previous=torch.zeros_like(q);beta=torch.zeros(1,device=b.device,dtype=b.dtype)
    basis=[q.clone()];diagonal=[];offdiagonal=[]
    for _ in range(iterations):
        z=action(q);alpha=(q*z).sum();z=z-alpha*q-beta*previous
        if reorthogonalize:
            for old in basis:z=z-(z*old).sum()*old
        beta=z.norm();diagonal.append(alpha);offdiagonal.append(beta)
        if float(beta)<=1e-30:break
        previous,q=q,z/beta;basis.append(q.clone())
    count=len(diagonal);matrix=torch.zeros(count,count,device=b.device,dtype=b.dtype)
    indices=torch.arange(count,device=b.device);matrix[indices,indices]=torch.stack(diagonal)
    if count>1:
        values=torch.stack(offdiagonal[:count-1]);matrix[indices[:-1],indices[1:]]=values;matrix[indices[1:],indices[:-1]]=values
    eigenvalues,eigenvectors=torch.linalg.eigh(matrix)
    if torch.any(eigenvalues < -1e-6):raise ValueError('Posterior covariance is not positive semidefinite')
    e1=torch.zeros(count,device=b.device,dtype=b.dtype);e1[0]=1
    coefficients=eigenvectors@(eigenvalues.clamp_min(0).sqrt()*(eigenvectors.T@e1))
    return (norm*(torch.stack(basis[:count],dim=1)@coefficients)).reshape(shape)

@register('posterior','ssd')
class ScaleSpacePosterior:
    def __init__(self,lanczos_iters=40,power_iters=20,safety_eps=.001):
        self.lanczos=lanczos_iters;self.power=power_iters;self.eps=safety_eps
    def __call__(self,state,clean_previous,alpha_s,alpha_t,sigma_s,sigma_t,operator,innovation,power_initial=None):
        # Source computes signal ratio in float64 NumPy then casts once.
        c=torch.as_tensor(float(alpha_t/alpha_s),device=state.device,dtype=state.dtype)
        mu=alpha_s*clean_previous
        ratio=torch.as_tensor(sigma_s,device=state.device,dtype=state.dtype)/torch.as_tensor(sigma_t,device=state.device,dtype=state.dtype)
        mean=mu+ratio**2*c*operator.transpose(state-c*operator(mu))
        if operator.source_hw==operator.target_hw:
            return mean,torch.as_tensor(sigma_s,device=state.device,dtype=state.dtype)*(1-ratio**2*c**2).sqrt()*innovation
        shape=(1,*innovation.shape[1:]);dimension=innovation[0].numel()
        def gram(v):
            z=v.view(shape);return (c*operator.transpose(c*operator(z))).reshape(-1)
        v=torch.randn(dimension,device=state.device,dtype=torch.float32) if power_initial is None else power_initial.clone()
        v=v/v.norm().clamp_min(1e-32);lmax=0.
        for _ in range(self.power):
            w=gram(v);lmax=float((v*w).sum().item());v=w/w.norm().clamp_min(1e-32)
        lmax=max(lmax,1e-12)
        # Native covariance routine squares sigma values before division; its
        # mean routine instead squares their ratio. Preserve both orders.
        sig_s=torch.as_tensor(sigma_s,device=state.device,dtype=state.dtype);sig_t=torch.as_tensor(sigma_t,device=state.device,dtype=state.dtype)
        rho=(sig_s**2)/(sig_t**2)
        effective=torch.minimum(rho,(1-self.eps)/torch.as_tensor(lmax,device=state.device,dtype=state.dtype))
        action=lambda v:v-effective*gram(v)
        outputs=[torch.as_tensor(sigma_s,device=state.device,dtype=state.dtype)*lanczos_sqrt(action,row,self.lanczos) for row in innovation]
        return mean,torch.stack(outputs)

@register('sampler','ssd')
class ScaleSpaceSampler:
    def __init__(self,resolutions,betas=None,unet_res_layer_map=None,clip_clean=True,lanczos_iters=40,power_iters=20,rescale_timesteps=False):
        from .batches import SSDBatch
        self.batch=SSDBatch(resolutions,betas,unet_res_layer_map,rescale_timesteps=rescale_timesteps);self.clip=clip_clean
        self.posterior=ScaleSpacePosterior(lanczos_iters,power_iters)
    @torch.no_grad()
    def __call__(self,model,initial,stream,model_kwargs=None,return_trajectory=False):
        from .adapters import SSDOutput
        state=initial;trajectory=[];path=self.batch.path
        for i in range(len(path.betas)-1,-1,-1):
            previous=max(i-1,0);pre,cond,rs,rt=self.batch.model_input(state,i)
            t=torch.full((len(state),),i,device=state.device,dtype=torch.long)
            network_time=t.float()*1000/len(path.betas) if self.batch.rescale else t
            clean=model(pre,network_time,cond_rs=cond,**(model_kwargs or {}))
            if rs!=rt and clean.shape[-1]!=rs:
                from .operators import resize
                clean=resize(clean,size=(rs,rs),mode='bilinear',antialias=True)
            if self.clip:clean=clean.clamp(-1,1)
            innovation=stream.normal(torch.empty(*state.shape[:-2],rs,rs,device=state.device,dtype=state.dtype))
            # Power iteration also draws noise. Route it through the supplied
            # stream, preserving the original draw order after innovation.
            power_initial=stream.normal(torch.empty(innovation[0].numel(),device=state.device,dtype=torch.float32)) if rs!=rt else None
            mean,noise=self.posterior(state,clean,path.alpha[previous],path.alpha[i],path.sigma[previous],path.sigma[i],self.batch.ops[i],innovation,power_initial=power_initial)
            state=mean+(i!=0)*noise;trajectory.append(state.clone())
        return (state,trajectory) if return_trajectory else state
