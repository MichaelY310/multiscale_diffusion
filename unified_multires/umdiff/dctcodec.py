"""DCTdiff's RGB/YCbCr token representation, independently organized."""
import numpy as np
import torch
from .registry import register

def zigzag(size):
    out=[]
    for diagonal in range(2*size-1):
        rows=list(range(max(0,diagonal-size+1),min(diagonal,size-1)+1))
        if diagonal%2==0:rows.reverse()
        out.extend(i*size+diagonal-i for i in rows)
    return np.array(out)

def frequency_weights(Y_std,Cb_std,Cr_std,block=4,frequencies=16):
    """Author EBFR fields (their statistics script computes entropy), not an
    independent standard-deviation estimate. Four Y blocks then Cb and Cr."""
    parts=[]
    for values in (Y_std,Cb_std,Cr_std):
        values=np.asarray(values)
        if values.size!=block*block or not 1<=frequencies<=values.size:raise ValueError('Invalid DCT frequency statistics')
        selected=values[zigzag(block)][:frequencies]
        if not np.isfinite(selected).all() or (selected<0).any() or selected.sum()<=0:raise ValueError('Frequency weights must be finite, nonnegative and nonzero')
        parts.append(torch.from_numpy(selected/(selected.sum()/len(selected))).float())
    return torch.cat([parts[0]]*4+parts[1:])

@register('boundary','dct_block_upsample')
class DCTBlockUpsample:
    """Released auxiliary RGB 4x4 -> 8x8 cosine-corrected block DCT lift."""
    def __call__(self,state,**kwargs):
        import cv2,math
        from . import native
        from .boundaries import BoundaryResult
        if state.dtype!=torch.uint8 or state.ndim!=4 or state.shape[1]!=3 or any(n%4 for n in state.shape[-2:]):raise ValueError('Auxiliary DCT upsampler requires BCHW RGB uint8 on grids divisible by four')
        env=native.extract('dctdiff/DCT_analysis_upsampling.py',['split_into_blocks','combine_blocks','dct_transform','idct_transform','DCT_upsampling'],dict(np=np,cv2=cv2,math=math))
        pixels=[env['DCT_upsampling'](x,block_sz=4) for x in state.cpu().permute(0,2,3,1).numpy()]
        return BoundaryResult(torch.from_numpy(np.stack(pixels)).permute(0,3,1,2).to(state.device),kwargs.get('time'),{'space':'RGB_uint8','backend':'unchanged author auxiliary analysis; nondifferentiable, not a latent diffusion bridge'})

@register('representation','dct_ycbcr')
class DCTYCbCr:
    def __init__(self,resolution=32,block=4,frequencies=16,bound=256.):
        if resolution%(2*block) or not 1<=frequencies<=block*block:raise ValueError('Invalid DCT layout')
        self.resolution=resolution;self.block=block;self.frequencies=frequencies;self.bound=np.array(bound)
        self.order=zigzag(block);self.reverse=np.argsort(self.order)
    def encode(self,images):
        """BCHW RGB uint8 (or normalized [-1,1] float) -> B,tokens,6F."""
        import cv2
        device=images.device
        data=images.detach().cpu()
        if data.dtype!=torch.uint8:data=((data+1)*127.5).round().clamp(0,255).byte()
        batches=[];s=self.block
        for image in data.permute(0,2,3,1).numpy():
            if image.shape!=(self.resolution,self.resolution,3):raise ValueError('DCT input resolution mismatch')
            r,g,b=(image[:,:,i] for i in range(3))
            y=.299*r+.587*g+.114*b
            cb=-.168736*r-.331264*g+.5*b+128
            cr=.5*r-.418688*g-.081312*b+128
            cb=cv2.resize(cb,(self.resolution//2,)*2,interpolation=cv2.INTER_LINEAR)
            cr=cv2.resize(cr,(self.resolution//2,)*2,interpolation=cv2.INTER_LINEAR)
            tokens=[]
            for row in range(0,self.resolution,2*s):
                for col in range(0,self.resolution,2*s):
                    blocks=[y[row+i*s:row+(i+1)*s,col+j*s:col+(j+1)*s] for i in range(2) for j in range(2)]
                    blocks.extend([p[row//2:row//2+s,col//2:col//2+s] for p in (cb,cr)])
                    coeff=np.array([cv2.dct(p.astype(np.float32)-128) for p in blocks]).reshape(6,s*s)
                    # Author's source assigns division back into float32 storage.
                    coeff[:4]=coeff[:4]/self.bound;coeff[4]=coeff[4]/self.bound;coeff[5]=coeff[5]/self.bound
                    tokens.append(coeff[:,self.order][:,:self.frequencies].reshape(-1))
            batches.append(np.array(tokens,dtype=np.float32))
        return torch.from_numpy(np.array(batches)).to(device)
    def decode(self,tokens,output_dtype='uint8'):
        import cv2
        if output_dtype not in ('uint8','float64'):raise ValueError('DCT decode output_dtype must be uint8 or float64')
        output=[];s=self.block;size=self.resolution
        for sample in tokens.detach().cpu().numpy():
            sample=np.clip(sample,-2,2).reshape(-1,6,self.frequencies)
            coeff=np.zeros((len(sample),6,s*s));coeff[:,:,:self.frequencies]=sample
            coeff=coeff[:,:,self.reverse]*self.bound
            y=np.zeros((size,size),np.float32);cb=np.zeros((size//2,size//2),np.float32);cr=np.zeros_like(cb)
            index=0
            for row in range(0,size,2*s):
                for col in range(0,size,2*s):
                    for i in range(2):
                        for j in range(2):y[row+i*s:row+(i+1)*s,col+j*s:col+(j+1)*s]=cv2.idct(coeff[index,2*i+j].reshape(s,s))+128
                    for k,p in enumerate((cb,cr)):p[row//2:row//2+s,col//2:col//2+s]=cv2.idct(coeff[index,4+k].reshape(s,s))+128
                    index+=1
            cb=cv2.resize(cb,(size,size),interpolation=cv2.INTER_LINEAR);cr=cv2.resize(cr,(size,size),interpolation=cv2.INTER_LINEAR)
            rgb=np.stack([y+1.402*(cr-128),y-.344136*(cb-128)-.714136*(cr-128),y+1.772*(cb-128)],axis=-1)
            # The author assembles clipped R/G/B into a float64 zeros array
            # before the final uint8 conversion. Expose that exact boundary.
            decoded=np.zeros((size,size,3),dtype=np.float64)
            decoded[:]=np.clip(rgb,0,255)
            output.append(decoded.astype(np.uint8) if output_dtype=='uint8' else decoded)
        # Keep uint8 pixels: no hidden float normalization/rounding in parity.
        return torch.from_numpy(np.array(output)).permute(0,3,1,2).to(tokens.device)


@register('representation','dct_ycbcr_grid')
class DCTYCbCrGrid(DCTYCbCr):
    """Same per-frame RGB codec, grid tokens for a jointly temporal backbone."""
    def encode(self,images):
        from .grids import as_video
        image=images.ndim==4; x=as_video(images); b,c,t,h,w=x.shape
        tokens=super().encode(x.permute(0,2,1,3,4).reshape(b*t,c,h,w))
        size=self.resolution//(2*self.block)
        out=tokens.transpose(1,2).reshape(b,t,6*self.frequencies,size,size).permute(0,2,1,3,4)
        return out.squeeze(2) if image else out

    def decode(self,x,output_dtype='uint8'):
        from .grids import as_video
        image=x.ndim==4; x=as_video(x); b,c,t,h,w=x.shape
        tokens=x.permute(0,2,3,4,1).reshape(b*t,h*w,c)
        out=super().decode(tokens,output_dtype).reshape(b,t,3,self.resolution,self.resolution).permute(0,2,1,3,4)
        return out.squeeze(2) if image else out
