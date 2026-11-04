"""Nested schedule and target operators that do not change sampling resolution."""
import torch
import torch.nn.functional as F

def snr_shift(gamma,scale=1.,power=1.):
    if scale<=1:return gamma
    snr=gamma/(1-gamma);scaled=snr/(scale**power)
    return 1/(1+1/scaled)

def dimr_targets(noise,levels=2,scale_correction=True):
    b,c,h,w=noise.shape;out={w:noise}
    for level in range(1,levels):
        ratio=2**level
        if h%2==0:
            if h%ratio or w%ratio:raise ValueError('Invalid even pyramid grid')
            kernel_size=ratio;padding=0
        else:
            if (h-1)%ratio or (w-1)%ratio:raise ValueError('Invalid odd pyramid grid')
            kernel_size=ratio+1;padding=kernel_size//2
        divisor=kernel_size if scale_correction else kernel_size**2
        kernel=torch.ones(c,1,kernel_size,kernel_size,device=noise.device)/divisor
        value=F.conv2d(noise,kernel,stride=ratio,padding=padding,groups=c)
        out[value.shape[-1]]=value
    return out

def mdm_posterior(state,prediction,gamma,gamma_previous,quantity='v',noise=None,eta=None,image_scale=1.,threshold='legacy'):
    """Released MDM DDPM/DDIM arithmetic (v/epsilon), with explicit noise."""
    a=gamma/gamma_previous;b=1-a;var=b*(1-gamma_previous)/(1-gamma)
    if quantity=='v':clean=state*gamma.sqrt()-prediction*(1-gamma).sqrt()
    elif quantity=='epsilon':clean=(state-prediction*(1-gamma).sqrt())/gamma.sqrt()
    else:raise ValueError('Unsupported MDM target')
    if threshold=='legacy':clean=clean.clamp(-image_scale,image_scale)/image_scale
    else:
        from .guidance import threshold_sample
        clean=threshold_sample(clean,threshold,image_scale)
    if eta is None:
        previous=clean*b*gamma_previous.sqrt()/(1-gamma)+state*a.sqrt()*(1-gamma_previous)/(1-gamma)
    else:
        eps=(state-clean*gamma.sqrt())/(1-gamma).sqrt()
        if eta>0:var=eta**2*var;previous=clean*gamma_previous.sqrt()+eps*(1-gamma_previous-var).sqrt()
        else:previous=clean*gamma_previous.sqrt()+eps*(1-gamma_previous).sqrt();noise=None
    if noise is not None:previous=previous+var.sqrt()*noise
    eps=(previous-gamma_previous.sqrt()*clean)/(1-gamma_previous).sqrt()
    return clean,previous,eps
