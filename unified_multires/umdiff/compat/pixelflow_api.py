"""The released class-conditional PixelFlow ndarray API over unified sampling."""
import torch
from ..samplers import PixelFlowSampler
from ..randomness import GlobalRandomStream

class PixelFlowPipeline:
    def __init__(self,transformer,num_stages=4,num_train_timesteps=1000):
        self.transformer=transformer;self.num_stages=num_stages;self.num_train_timesteps=num_train_timesteps

    @torch.no_grad()
    def __call__(self,prompt,height,width,num_inference_steps=30,guidance_scale=4.,num_images_per_prompt=1,device=None,shift=1.,use_ode_dopri5=False):
        if num_images_per_prompt!=1:raise ValueError('The released class-label packing does not repeat labels for multiple images; use repeated prompt labels explicitly')
        if not prompt or any(not isinstance(x,int) for x in prompt):raise ValueError('The released class-conditional API requires a list of integer labels')
        device=next(self.transformer.parameters()).device if device is None else torch.device(device)
        factor=2**(self.num_stages-1)
        initial=torch.randn(len(prompt),3,height//factor,width//factor,device=device,dtype=torch.float32)
        sampler=PixelFlowSampler(stages=self.num_stages,train_steps=self.num_train_timesteps,steps=num_inference_steps,shift=shift,guidance_scale=guidance_scale,solver='native_dopri5' if use_ode_dopri5 else 'euler')
        labels=torch.tensor(prompt,dtype=torch.int32,device=device)
        value=sampler(self.transformer,initial,GlobalRandomStream(),{'class_labels':labels})
        return (value/2+.5).clamp(0,1).cpu().permute(0,2,3,1).float().numpy()
