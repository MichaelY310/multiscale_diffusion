"""Independent equation-level references; these are NOT author repositories.

No production process, clock, boundary, solver or wavelet helper is imported.
Floating point operation ordering is specified so a matched numerical policy
can be compared byte-for-byte, separately from paper/implementation provenance.
"""
import math
import torch
import torch.nn.functional as F


def up(x):
    return x.repeat_interleave(2,2).repeat_interleave(2,3)


def project(x):
    b,c,h,w=x.shape
    return up(x.reshape(b,c,h//2,2,w//2,2).mean((3,5)))


def fdm(config,model,noise,labels):
    s=config['process']['args'];grids=config['grids'];bounds=s['bounds']
    terminal=s.get('terminal',.9999);eta=s.get('eta',1.)
    def coefficients(t,k):
        t=t.reshape(-1,1,1,1)
        alpha=torch.cos(math.pi*t/2)
        sigma=torch.sin(math.pi*t/2)/(grids[-1][-1]/grids[k][-1])
        if s.get('rescaling','vp')=='vp':
            norm=(alpha.square()+sigma.square()).sqrt()
            alpha,sigma=alpha/norm,sigma/norm
        return alpha,sigma
    x=noise*coefficients(noise.new_tensor(terminal),0)[1];trace=[x.clone()]
    for k,n in enumerate(s['steps']):
        times=torch.linspace(min(bounds[k],terminal),bounds[k+1],n+1)
        for t,r in zip(times[:-1],times[1:]):
            alpha,sigma=coefficients(t.expand(len(x)),k)
            alpha_next,sigma_next=coefficients(r.expand(len(x)),k)
            pred=model(x,t.expand(len(x)),y=labels)
            if s.get('prediction','epsilon')=='epsilon':
                eps=pred['epsilon'];clean=(x-sigma*eps)/alpha
            else:
                clean=pred['signal'];eps=(x-alpha*clean)/sigma
            # Algorithm1 degradation correction during this reverse interval.
            endpoint=clean+pred['residual']*((t-r)/(t-bounds[k+1]))
            q=(sigma.square()-(alpha/alpha_next).square()*sigma_next.square()).clamp_min(0)
            variance=eta**2*sigma_next.square()*q/sigma.square()
            x=alpha_next*endpoint+(sigma_next.square()-variance).clamp_min(0).sqrt()*eps
            if eta and r>0:x=x+variance.sqrt()*torch.randn_like(x)
            trace.append(x.clone())
        if k+1<len(grids):
            residual_noise=(x-alpha_next*endpoint)/sigma_next
            fine=F.interpolate(endpoint,size=grids[k+1],mode='bilinear',align_corners=False)
            z=torch.randn_like(fine)
            coupled=up(residual_noise)/2+z-project(z)
            alpha_next,sigma_next=coefficients(x.new_full((len(x),),bounds[k+1]),k+1)
            x=alpha_next*fine+sigma_next*coupled;trace.append(x.clone())
    return x,trace


def pyramidal_wan_path(config,model,noise,labels):
    settings=config['process']['args'];sizes=config['grids'];bounds=settings['natural_bounds']
    x=noise.clone();trace=[x.clone()]
    def sigma(u,k):
        scale=x.new_tensor(sizes[-1][-1]/sizes[k][-1])
        return u/(scale-(scale-1)*u)
    for k,n in enumerate(settings['steps']):
        times=torch.linspace(bounds[k],bounds[k+1],n+1)
        for t,r in zip(times[:-1],times[1:]):
            a,b=sigma(t,k),sigma(r,k)
            velocity=model(x,a.reshape(-1).expand(len(x)),y=labels)
            x=x+(b-a)*velocity;trace.append(x.clone())
        if k+1<len(sizes):
            lifted=up(x);z=torch.randn_like(lifted);s=sigma(x.new_tensor(bounds[k+1]),k)
            omega=x.new_tensor(2.)
            x=(lifted+omega*s*(z-project(z)))/(1+(omega-1)*s)
            trace.append(x.clone())
    return x,trace


def haar_decode(parts):
    # Inverse separable 2D orthonormal Haar (width then height).
    x=parts[0]
    for band in parts[1:]:
        leaves=[x,*band.chunk(3,1)]
        a,b,c,d=leaves
        lo=torch.empty(*a.shape[:-1],a.shape[-1]*2,dtype=a.dtype)
        hi=torch.empty_like(lo)
        lo[...,0::2]=(a+b)/math.sqrt(2);lo[...,1::2]=(a-b)/math.sqrt(2)
        hi[...,0::2]=(c+d)/math.sqrt(2);hi[...,1::2]=(c-d)/math.sqrt(2)
        x=torch.empty(*lo.shape[:2],lo.shape[2]*2,lo.shape[3],dtype=lo.dtype)
        x[:,:,0::2]=(lo+hi)/math.sqrt(2);x[:,:,1::2]=(lo-hi)/math.sqrt(2)
    return x


def nonuniform_haar(config,model,noise,labels):
    settings=config['process']['args'];sizes=config['grids'];levels=len(sizes)
    high_snr=settings.get('logsnr_max',9.21);low_snr=settings.get('logsnr_min',-10.)
    # These numerical endpoints are the selected finite-SNR implementation,
    # not a claim of an author script or the exact infinite-SNR limit.
    terminal=settings.get('terminal',1e-5);stochastic=settings.get('sampling','sde')=='sde'
    state={0:noise.clone()};trace=[{0:noise.clone()}]
    for k,n in enumerate(settings['steps']):
        high=1.-k/levels;low=1.-(k+1)/levels if k+1<levels else terminal
        if k:state[k]=torch.randn(len(noise),noise.shape[1]*3,*sizes[k-1])
        times=torch.linspace(high,low,n+1)
        for t,r in zip(times[:-1],times[1:]):
            clocks=t.expand(len(noise))[:,None].expand(-1,k+1)
            pred=model(state,clocks,y=labels)
            for j in state:
                slope=(low_snr-high_snr)/(1.-j/levels)
                logsnr=high_snr+slope*t.expand(len(noise)).reshape(-1,1,1,1)
                abar=logsnr.sigmoid();sigma=(1-abar).sqrt();beta=-slope*(1-abar)
                score=-pred[j]/sigma
                drift=-.5*beta*state[j]-beta*score*(1. if stochastic else .5)
                state[j]=state[j]+(r-t)*drift
                if stochastic:state[j]=state[j]+(beta*(t-r)).sqrt()*torch.randn_like(state[j])
            trace.append({j:v.clone() for j,v in state.items()})
    return haar_decode([state[j] for j in range(levels)]),trace


def flow(config,model,noise,labels):
    x=noise.clone();trace=[x.clone()];settings=config['clock']['args']
    times=torch.linspace(settings['start'],settings['end'],251)
    for t,r in zip(times[:-1],times[1:]):
        velocity=model(x,t.expand(len(x)),y=labels)
        x=x+(r-t)*velocity;trace.append(x.clone())
    return x,trace
