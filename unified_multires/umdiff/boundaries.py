"""Stage kernels: transform state, preserve/replace noise, and change clocks."""
from dataclasses import dataclass,field
import math
import numpy as np
import torch
from .registry import register
from .operators import repeat,Projection,channel_up,resize
from .noise import coupled_lift,equicorrelated_blocks

@dataclass
class BoundaryResult:
    state:torch.Tensor
    time:object=None
    metadata:dict=field(default_factory=dict)

@register('boundary','identity')
class Identity:
    def __call__(self,state,**kwargs):return BoundaryResult(state,kwargs.get('time'))

@register('boundary','resize')
class Resize:
    def __init__(self,factor=2,mode='nearest',antialias=False):self.factor=factor;self.mode=mode;self.antialias=antialias
    def __call__(self,state,**kwargs):return BoundaryResult(resize(state,scale=self.factor,mode=self.mode,antialias=self.antialias),kwargs.get('time'))

@register('boundary','rdd_refresh')
@register('boundary','rdd_fresh')
class RDDFresh:
    def __init__(self,factor=2):self.factor=factor
    def __call__(self,state,*,clean,alpha,sigma,noise,**kwargs):
        return BoundaryResult(alpha*repeat(clean,self.factor)+sigma*noise,kwargs.get('time'),{'noise_coupling':'fresh_iid'})

@register('boundary','rdd_noise_filling')
class RDDNoiseFilling:
    """Recover endpoint residual, lift isometrically, add null-space innovation.

    previous_alpha/sigma belong to the OLD noisy endpoint; alpha/sigma to the
    NEW stage. Identity Gaussian covariance requires an ideal coarse residual,
    and is not an assertion about arbitrary model prediction errors.
    """
    def __init__(self,factor=2):self.factor=factor
    def __call__(self,state,*,clean,alpha,sigma,previous_alpha,previous_sigma,noise,**kwargs):
        if torch.any(torch.as_tensor(previous_sigma)<=0):raise ValueError('Cannot recover residual at sigma=0')
        recovered=(state-previous_alpha*clean)/previous_sigma
        op=Projection(factor=self.factor)
        eps=repeat(recovered,self.factor)/float(self.factor)+(noise-op(noise))
        return BoundaryResult(alpha*repeat(clean,self.factor)+sigma*eps,kwargs.get('time'),
                              {'noise_coupling':'recovered_endpoint','fine_noise':eps})

@register('boundary','isometric')
class Isometric:
    def __init__(self,factor=2,rho=1.):self.factor=factor;self.rho=rho
    def __call__(self,state,*,clean,alpha,sigma,coarse_noise,low_innovation,high_innovation,**kwargs):
        eps=coupled_lift(coarse_noise,low_innovation,high_innovation,self.factor,self.rho)
        return BoundaryResult(alpha*repeat(clean,self.factor)+sigma*eps,kwargs.get('time'),{'fine_noise':eps})

@register('boundary','rdd_legacy_posterior')
class RDDLegacyPosterior:
    """Dense historical RDD posterior, including its block-vector convention.

    This is distinct from the current image entrypoint's fresh boundary.
    Preserve source factor-two geometry and operation order for oracle parity.
    """
    def __init__(self,factor=2):
        if factor!=2:raise ValueError('Historical RDD vector_to_image hardcodes scale two')
        self.factor=factor
    def __call__(self,state,*,clean,alpha,sigma=None,variance=None,stream,**kwargs):
        from .tree import expand
        batch,channels,h,w=state.shape;n=h*w;m=4*n
        a=expand(alpha,state).expand_as(state)[:,:,0,0,None,None]
        if variance is None:
            if sigma is None:raise ValueError('Supply sigma or explicit native variance')
            b=expand(sigma,state).expand_as(state)[:,:,0,0,None,None].square()
        else:b=expand(variance,state).expand_as(state)[:,:,0,0,None,None]
        if torch.any(b<=0):raise ValueError('Posterior boundary requires positive noise variance')
        eye=torch.eye(n,device=state.device,dtype=state.dtype)
        D=eye.repeat_interleave(4,dim=1)/4;U=4*D.T
        complement=eye-D@D.T
        B_inv=torch.linalg.inv(b*complement)
        precision=torch.eye(m,device=state.device,dtype=state.dtype)+D.T@torch.linalg.inv(complement)@D
        covariance=b*torch.linalg.inv(precision)[None,None]
        dt_binv=(B_inv.permute(0,1,3,2)@D).permute(0,1,3,2)
        mu1=torch.einsum('n c i j,n c j->n c i',dt_binv,state.flatten(2))
        mu2=(a/b).squeeze(-1)*torch.einsum('i j,n c j->n c i',U,clean.flatten(2))
        mean=torch.einsum('n c i j,n c j->n c i',covariance,mu1+mu2)
        sample=mean+(torch.linalg.cholesky(covariance)@stream.normal(mean)[...,None]).squeeze(-1)
        def image(x):return x.reshape(batch,channels,h,w,2,2).permute(0,1,2,4,3,5).reshape(batch,channels,2*h,2*w)
        return BoundaryResult(image(sample),kwargs.get('time'),{'mean':image(mean),'covariance':covariance,'variant':'historical_dense_posterior'})

@register('boundary','gaussian_conditional')
class GaussianConditional:
    """General dense linear-Gaussian lift for user-defined D, covariance and R.

    Given X~N(m,C), observe Y=D X+eta, eta~N(0,R). Runtime tensors make
    operator normalization and nonorthogonal interpolation explicit.
    """
    def __call__(self,state,*,mean,covariance,observation,observation_noise=None,noise,**kwargs):
        D=observation;C=covariance;R=observation_noise
        S=D@C@D.T
        if R is not None:S=S+R
        gain=torch.linalg.solve(S,D@C).T
        mu=mean.flatten(2)+(state.flatten(2)-mean.flatten(2)@D.T)@gain.T
        conditional=C-gain@D@C;conditional=(conditional+conditional.T)/2
        values,vectors=torch.linalg.eigh(conditional)
        tolerance=100*torch.finfo(C.dtype).eps*C.abs().max().clamp_min(1.)
        if values.min() < -tolerance:raise ValueError('Conditional covariance is not positive semidefinite')
        root=vectors*values.clamp_min(0).sqrt()[None,:]
        sample=mu+noise.flatten(2)@root.T
        return BoundaryResult(sample.reshape_as(mean),kwargs.get('time'),{'mean':mu.reshape_as(mean),'covariance':conditional})

@register('boundary','pixelflow')
class PixelFlowBoundary:
    def __init__(self,gamma=-1/3,jitter=1e-6):
        if gamma>=0:raise ValueError('PixelFlow requires negative gamma')
        self.gamma=gamma;self.jitter=jitter
    def __call__(self,state,*,time,noise=None,standard=None,**kwargs):
        lifted=repeat(state,2)
        if noise is None:noise=equicorrelated_blocks(lifted,self.gamma,self.jitter,standard)
        a=1/(math.sqrt(1-1/self.gamma)*(1-time)+time)
        b=a*(1-time)/math.sqrt(-self.gamma)
        return BoundaryResult(a*lifted+b*noise,time*a,{'state_scale':a,'noise_scale':b})

@register('boundary','permutation')
class PermutationBoundary:
    def __init__(self,factor=2,width_only=False):self.factor=factor;self.width_only=width_only
    def __call__(self,state,**kwargs):return BoundaryResult(channel_up(state,self.factor,self.width_only),kwargs.get('time'))

@register('boundary','learned_affine')
class LearnedAffine(torch.nn.Module):
    """NAMI BridgeFlow. Explicit parameters; never loads pretrained weights."""
    def __init__(self,shapes=((16,64,64),(16,128,128))):
        super().__init__();self.scale=torch.nn.Parameter(torch.ones(len(shapes)))
        self.shift=torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(1,*s)) for s in shapes])
    def forward(self,state,*,stage=1,**kwargs):
        idx=stage-1
        out=repeat(state,2)*self.scale[idx]+self.shift[idx]
        return BoundaryResult(out,kwargs.get('time'))

@register('boundary','spectral')
class SpectralBoundary:
    """Independent SPEED port, retaining author's NumPy RNG and fp32 policy."""
    def __init__(self,transform='dct',factor=2):
        if transform not in ('dct','dwt','fft'):raise ValueError('Invalid basis')
        self.transform=transform;self.factor=factor
    def __call__(self,state,*,time,seed=0,target_hw=None,ratio=None,**kwargs):
        from scipy.fft import dctn,idctn
        import pywt
        array=state.detach().cpu().float().numpy()
        h,w=array.shape[-2:];oh,ow=target_hw or (h*self.factor,w*self.factor)
        ratio=ratio or self.factor
        if oh<h or ow<w:raise ValueError('Cannot spectrally shrink')
        if self.transform=='dwt' and (oh,ow)!=(2*h,2*w):raise ValueError('Haar expansion requires factor two')
        rng=np.random.default_rng(seed)
        out=np.empty(array.shape[:-2]+(oh,ow),dtype=np.float32)
        for index in np.ndindex(array.shape[:-2]):
            x=array[index]
            if self.transform=='dct':
                spectrum=time*rng.standard_normal((oh,ow)).astype(np.float32)
                spectrum[:h,:w]=dctn(x,type=2,norm='ortho')
                y=idctn(spectrum,type=2,norm='ortho')
            elif self.transform=='dwt':
                bands=tuple(time*rng.standard_normal((h,w)).astype(np.float32) for _ in range(3))
                y=pywt.waverec2([x,bands],'haar',mode='periodization')
            else:
                nr=rng.standard_normal((oh,ow)).astype(np.float32)
                ni=rng.standard_normal((oh,ow)).astype(np.float32)
                spectrum=np.fft.fftshift(time*(nr+1j*ni))
                top,left=(oh-h)//2,(ow-w)//2
                spectrum[top:top+h,left:left+w]=np.fft.fftshift(np.fft.fft2(x,norm='ortho'))
                y=np.fft.ifft2(np.fft.ifftshift(spectrum),norm='ortho').real
            out[index]=y.astype(np.float32)
        scale=ratio/(1+(ratio-1)*time)
        expanded=torch.from_numpy((scale*out).astype(np.float32)).to(device=state.device,dtype=state.dtype)
        return BoundaryResult(expanded,time*scale,{'state_scale':scale,'time_orientation':'data_to_noise'})

@register('boundary','codec_refine')
class CodecRefine:
    """MrFlow orchestration with explicitly supplied decoder/SR/encoder.

    Pixel/PIL conversion belongs to these supplied callables. No implicit model
    download or identity replacement for RealESRGAN is allowed.
    """
    def __init__(self,decoder,super_resolution,encoder,first_sigma=.12):
        self.decoder=decoder;self.sr=super_resolution;self.encoder=encoder;self.sigma=first_sigma
    def __call__(self,state,*,noise,**kwargs):
        x=self.encoder(self.sr(self.decoder(state)))
        return BoundaryResult((1-self.sigma)*x+self.sigma*noise,self.sigma)
