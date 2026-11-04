"""Real-ESRGAN PIL/tile boundary, with an explicitly supplied SR network.

Matches ai-forever/Real-ESRGAN at MrFlow's pinned gitlink. No downloads.
"""
import numpy as np
import torch
from PIL import Image
from .registry import register

@register('boundary','real_esrgan_pixels')
class RealESRGANPixels:
    def __init__(self,network,scale=2,batch_size=4,patch_size=192,padding=24,pad_size=15):
        self.network=network;self.scale=scale;self.batch=batch_size;self.patch=patch_size;self.padding=padding;self.pad=pad_size
        if min(scale,batch_size,patch_size,padding,pad_size)<=0:raise ValueError('Native tiled SR requires positive dimensions')
    @torch.no_grad()
    def __call__(self,image):
        x=np.asarray(image.convert('RGB'));h,w,c=x.shape;p=self.pad
        if min(h,w)<p:raise ValueError('Image is smaller than native symmetric padding')
        # Author includes the edge itself; NumPy reflect mode would exclude it.
        x=np.pad(x,((p,p),(p,p),(0,0)),mode='symmetric')
        height,width=x.shape[:2];ps=self.patch;op=self.padding
        extended=np.pad(x,((0,(-height)%ps),(0,(-width)%ps),(0,0)),mode='edge')
        padded=np.pad(extended,((op,op),(op,op),(0,0)),mode='edge')
        patches=np.array([padded[i:i+ps+2*op,j:j+ps+2*op] for i in range(0,extended.shape[0],ps) for j in range(0,extended.shape[1],ps)])
        parameter=next(self.network.parameters());inputs=torch.FloatTensor(patches/255).permute(0,3,1,2).to(parameter.device)
        result=torch.cat([self.network(inputs[i:i+self.batch]) for i in range(0,len(inputs),self.batch)])
        result=result.permute(0,2,3,1).clamp(0,1).cpu().numpy();q=op*self.scale
        result=result[:,q:-q,q:-q,:]
        shape=(padded.shape[0]*self.scale,padded.shape[1]*self.scale,3);canvas=np.zeros(shape)
        stride=ps*self.scale;per_row=shape[1]//stride
        for i,patch in enumerate(result):
            row,col=divmod(i,per_row);canvas[row*stride:(row+1)*stride,col*stride:(col+1)*stride]=patch
        canvas=(canvas[:height*self.scale,:width*self.scale]*255).astype(np.uint8);q=p*self.scale
        return Image.fromarray(canvas[q:-q,q:-q])

@register('workflow','mrflow')
class MrFlow:
    """Plug supplied native base/refine pipelines and SR into the staged method."""
    def __init__(self,low_size=512,high_size=1024,low_steps=12,refine_steps=1,refine_sigma=.12,guidance_scale=3.5,seed=2026):
        self.low=low_size;self.high=high_size;self.steps=low_steps;self.refine_steps=refine_steps;self.sigma=refine_sigma;self.guidance=guidance_scale;self.seed=seed
    @torch.no_grad()
    def __call__(self,base,super_resolution,refiner,prompt_embeds=None,pooled_prompt_embeds=None,conditioning=None):
        from . import native
        direct=native.load('mrflow/mrflow_utils.py').direct_sigma_schedule
        device=next(base.transformer.parameters()).device
        if conditioning is None:
            kwargs=dict(prompt=None,prompt_embeds=prompt_embeds,pooled_prompt_embeds=pooled_prompt_embeds,guidance_scale=self.guidance)
        else:
            # Qwen's masks/negative embeddings/true_cfg_scale belong to the
            # supplied pipeline interface. Keep them intact across both stages.
            kwargs=dict(conditioning)
            if prompt_embeds is not None or pooled_prompt_embeds is not None:
                raise ValueError('Use either explicit pipeline conditioning or Flux embeddings')
        low=base(**kwargs,width=self.low,height=self.low,num_inference_steps=self.steps,generator=torch.Generator(device=device).manual_seed(self.seed)).images[0]
        up=super_resolution(low).resize((self.high,self.high))
        with direct(refiner.scheduler,self.sigma,self.refine_steps,device):
            refined=refiner(**kwargs,image=up,width=self.high,height=self.high,strength=1.,num_inference_steps=self.refine_steps,generator=torch.Generator(device=device).manual_seed(self.seed+1)).images[0]
        return {'low':low,'upscaled':up,'refined':refined}
