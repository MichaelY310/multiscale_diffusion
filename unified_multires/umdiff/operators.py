"""Independent signal, noise and representation operators."""
import math
import torch
import torch.nn.functional as F
from .registry import register

def resize(x,size=None,scale=None,mode='bilinear',antialias=False,align_corners=False):
    if x.ndim==5 and (size is None or isinstance(size,int) or len(size)==2) and (scale is None or isinstance(scale,(int,float))):
        b,c,t,h,w=x.shape
        flat=x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        out=resize(flat,size,scale,mode,antialias,align_corners)
        return out.reshape(b,t,c,*out.shape[-2:]).permute(0,2,1,3,4)
    kw={'size':size,'scale_factor':scale,'mode':mode}
    if mode in ('bilinear','bicubic','linear','trilinear'):
        kw['align_corners']=align_corners
    if mode in ('bilinear','bicubic'):
        kw['antialias']=antialias
    return F.interpolate(x,**kw)

def block_average(x,factor=2):
    if x.ndim==5:
        from .grids import block_down
        return block_down(x,(1,factor,factor))
    b,c,h,w=x.shape
    if h%factor or w%factor:
        raise ValueError('Block projection needs divisible dimensions')
    return x.reshape(b,c,h//factor,factor,w//factor,factor).mean((3,5))

def repeat(x,factor=2):
    return x.repeat_interleave(factor,-1).repeat_interleave(factor,-2)

@register('operator','projection')
class Projection:
    def __init__(self,factor=2,down='average',up='nearest',antialias=False):
        self.factor=factor;self.down_mode=down;self.up_mode=up;self.antialias=antialias
    def down(self,x):
        if self.down_mode=='average':
            return block_average(x,self.factor)
        return resize(x,scale=1/self.factor,mode=self.down_mode,antialias=self.antialias)
    def up(self,x):
        if self.up_mode=='nearest':return repeat(x,self.factor)
        return resize(x,scale=self.factor,mode=self.up_mode,antialias=self.antialias)
    def __call__(self,x):return self.up(self.down(x))
    def high(self,x):return x-self(x)
    @property
    def orthogonal(self):return self.down_mode=='average' and self.up_mode=='nearest'

class BilinearOperator:
    def __init__(self,source_hw,target_hw,antialias=True,align_corners=False):
        self.source_hw=tuple(source_hw);self.target_hw=tuple(target_hw)
        self.antialias=antialias;self.align_corners=align_corners
    def __call__(self,x):
        if x.shape[-2:]!=self.source_hw:raise ValueError('Wrong resize source shape')
        if self.source_hw==self.target_hw:return x
        return resize(x,size=self.target_hw,antialias=self.antialias,align_corners=self.align_corners)
    def transpose(self,y,create_graph=False):
        if y.shape[-2:]!=self.target_hw:raise ValueError('Wrong adjoint input shape')
        if self.source_hw==self.target_hw:return y
        with torch.enable_grad():
            dummy=torch.zeros((*y.shape[:-2],*self.source_hw),device=y.device,dtype=y.dtype,requires_grad=True)
            return torch.autograd.grad(self(dummy),dummy,y,create_graph=create_graph)[0]

def dct_matrix(n,device,dtype):
    k=torch.arange(n,device=device,dtype=dtype)[:,None]
    j=torch.arange(n,device=device,dtype=dtype)[None,:]
    m=torch.cos(math.pi/n*(j+.5)*k)*math.sqrt(2/n)
    m[0]*=math.sqrt(.5)
    return m

def dct2(x,inverse=False):
    h,w=x.shape[-2:]
    a=dct_matrix(h,x.device,x.dtype);b=dct_matrix(w,x.device,x.dtype)
    return a.T@x@b if inverse else a@x@b.T

def dct1_fft(x,inverse=False):
    """Orthonormal DCT-II/III with native Relay FFT arithmetic order.

    Algorithm attributed to zh217/torch-dct, as used in pinned Relay blurring.py.
    Matrix and FFT backends are algebraically equal but Adam can amplify their
    near-zero-gradient rounding differences. Keep the choice explicit.
    """
    shape=x.shape;n=shape[-1];flat=x.contiguous().view(-1,n)
    if not inverse:
        vector=torch.cat([flat[:,::2],flat[:,1::2].flip([1])],1)
        spectrum=torch.view_as_real(torch.fft.fft(vector,dim=1))
        theta=-torch.arange(n,dtype=x.dtype,device=x.device)[None,:]*math.pi/(2*n)
        result=spectrum[:,:,0]*torch.cos(theta)-spectrum[:,:,1]*torch.sin(theta)
        result[:,0]/=math.sqrt(n)*2;result[:,1:]/=math.sqrt(n/2)*2
        return 2*result.view(shape)
    values=flat/2
    values[:,0]*=math.sqrt(n)*2;values[:,1:]*=math.sqrt(n/2)*2
    theta=torch.arange(n,dtype=x.dtype,device=x.device)[None,:]*math.pi/(2*n)
    imaginary=torch.cat([values[:,:1]*0,-values.flip([1])[:,:-1]],1)
    real=values*torch.cos(theta)-imaginary*torch.sin(theta)
    imag=values*torch.sin(theta)+imaginary*torch.cos(theta)
    spectrum=torch.cat([real.unsqueeze(2),imag.unsqueeze(2)],2)
    vector=torch.fft.irfft(torch.view_as_complex(spectrum),n=spectrum.shape[1],dim=1)
    result=vector.new_zeros(vector.shape)
    result[:,::2]+=vector[:,:n-n//2];result[:,1::2]+=vector.flip([1])[:,:n//2]
    return result.view(shape)

def block_dct(x,patch,inverse=False,backend='fft'):
    if x.ndim==5:
        b,c,t,h,w=x.shape
        flat=x.permute(0,2,1,3,4).reshape(b*t,c,h,w)
        out=block_dct(flat,patch,inverse,backend)
        return out.reshape(b,t,c,h,w).permute(0,2,1,3,4)
    b,c,h,w=x.shape
    if h%patch or w%patch:raise ValueError('DCT block size must divide image')
    z=x.reshape(b,c,h//patch,patch,w//patch,patch).permute(0,1,2,4,3,5)
    if backend=='matrix':z=dct2(z,inverse)
    elif backend=='fft':z=dct1_fft(dct1_fft(z,inverse).transpose(-1,-2),inverse).transpose(-1,-2)
    else:raise ValueError('Unknown DCT backend')
    return z.permute(0,1,2,4,3,5).reshape(b,c,h,w)

def heat_blur(x,sigma,patch=4,min_scale=.001,backend='fft'):
    frequencies=torch.pi*torch.linspace(0,patch-1,patch,device=x.device)/patch
    frequency2=frequencies[:,None]**2+frequencies[None,:]**2
    damping=torch.exp(-frequency2.repeat(x.shape[-2]//patch,x.shape[-1]//patch)*(sigma**2/2))
    return block_dct(block_dct(x,patch,backend=backend)*(damping*(1-min_scale)+min_scale),patch,True,backend=backend)

def rolling_noise(eps,block=4):
    # Grouped circular convolution, equal native cyclic roll sum and variance.
    out=torch.zeros_like(eps)
    for i in range(block):
        for j in range(block):out=out+torch.roll(eps,(i,j),(-2,-1))
    return out/block

def channel_down(x,factor=2,width_only=False):
    if x.ndim==5:
        b,c,t,h,w=x.shape
        out=channel_down(x.permute(0,2,1,3,4).reshape(b*t,c,h,w),factor,width_only)
        return out.reshape(b,t,*out.shape[1:]).permute(0,2,1,3,4)
    b,c,h,w=x.shape
    if width_only:
        return x.reshape(b,c,h,w//factor,factor).permute(0,1,4,2,3).reshape(b,c*factor,h,w//factor)
    return x.reshape(b,c,h//factor,factor,w//factor,factor).permute(0,1,3,5,2,4).reshape(b,c*factor**2,h//factor,w//factor)

def channel_up(x,factor=2,width_only=False):
    if x.ndim==5:
        b,c,t,h,w=x.shape
        out=channel_up(x.permute(0,2,1,3,4).reshape(b*t,c,h,w),factor,width_only)
        return out.reshape(b,t,*out.shape[1:]).permute(0,2,1,3,4)
    b,c,h,w=x.shape
    if width_only:
        return x.reshape(b,c//factor,factor,h,w).permute(0,1,3,4,2).reshape(b,c//factor,h,w*factor)
    return x.reshape(b,c//factor**2,factor,factor,h,w).permute(0,1,4,2,5,3).reshape(b,c//factor**2,h*factor,w*factor)
