"""TRAINING-REQUIRED clean-parent renderer; not native-VAE-equivalent.

Fresh detail drives flow and unrestricted RGB correction. Parent pixels are
sampled directly, not averaged into a small vector. This codec by itself is
generic; only early committed-parent scheduling is process-dependent.
"""
import torch
from torch import nn
import torch.nn.functional as F


class LocalBlock(nn.Module):
    def __init__(self,width):
        super().__init__()
        self.depthwise=nn.Conv3d(width,width,3,padding=1,groups=width)
        self.project=nn.Conv3d(width,width,1)

    def forward(self,x):
        return x+self.project(F.silu(self.depthwise(x)))


class TemporalInnovationRenderer(nn.Module):
    """4n+1-aware output shape, BUT time-basis fidelity still needs training.

    Parent/detail have the same latent shape [B,16,T,H,W]. Fine reconstructed
    latent has2T frames, hence the native Wan RGB length is8T-3. The ordinary
    counterpart can use final-pooled c/h; it lacks early final-parent access.
    """
    def __init__(self,width=64,depth=4,forecast_correction=False):
        super().__init__()
        self.forecast_correction=forecast_correction
        self.input=nn.Conv3d(48 if forecast_correction else 32,width,1)
        self.blocks=nn.Sequential(*(LocalBlock(width) for _ in range(depth)))
        self.rgb=nn.Conv3d(width,3*8*8*8,1)
        self.flow=nn.Conv3d(width,3*8,1)
        # Small nonzero corrections: test h dependence, not native-quality.
        nn.init.normal_(self.rgb.weight,std=.001)
        nn.init.normal_(self.flow.weight,std=.001)
        nn.init.zeros_(self.rgb.bias)
        nn.init.zeros_(self.flow.bias)

    def prepare_detail(self,parent,detail,reference_parent=None):
        if parent.shape!=detail.shape or parent.shape[1]!=16:
            raise ValueError('Matched16-channel parent/detail required')
        inputs=[parent,detail]
        if self.forecast_correction:
            if reference_parent is None or reference_parent.shape!=parent.shape:
                raise ValueError('Forecast control requires matched reference parent')
            # Strong ordinary control: its earlier clean forecast may be wrong.
            # It gets the true final low band AND its complete change, not just
            # stale RGB. Committed runs use the SAME model with delta_c=0.
            inputs.append(parent-reference_parent)
        elif reference_parent is not None:
            raise ValueError('Enable forecast_correction to use reference_parent')
        x=self.blocks(F.silu(self.input(torch.cat(inputs,1))))
        b,_,t,h,w=x.shape
        # Do not gate the expensive feature computation on RGB availability:
        # the strongest post-final baseline may run this while Dc executes.
        correction=self.rgb(x).reshape(b,3,8,8,8,t,h,w)
        correction=correction.permute(0,1,5,2,6,3,7,4).reshape(b,3,8*t,8*h,8*w)[:,:,3:]
        flow=self.flow(x).reshape(b,3,8,t,h,w).permute(0,1,3,2,4,5).reshape(b,3,8*t,h,w)[:,:,3:]
        return correction,flow

    def finish(self,parent_rgb,prepared,backend='torch'):
        if backend=='triton':
            if __package__:
                from .progressive_render_triton import fused_render
            else:
                from progressive_render_triton import fused_render
            return fused_render(parent_rgb,prepared)
        if backend!='torch':
            raise ValueError('Backend must be torch or triton')
        correction,flow=prepared
        b,_,t,h,w=correction.shape
        if parent_rgb.ndim==4:
            parent_rgb=parent_rgb.unsqueeze(0)
        if parent_rgb.shape[:2]!=(b,3):
            raise ValueError('Parent RGB shape mismatch')
        offsets=F.interpolate(flow.float(),size=(t,h,w),mode='trilinear',align_corners=True).tanh()
        # Dense coordinates are necessary output work, not a new denoiser.
        z=torch.linspace(-1,1,t,device=flow.device)
        y=torch.linspace(-1,1,h,device=flow.device)
        x=torch.linspace(-1,1,w,device=flow.device)
        zz,yy,xx=torch.meshgrid(z,y,x,indexing='ij')
        grid=torch.stack((xx,yy,zz),-1).unsqueeze(0).expand(b,-1,-1,-1,-1).clone()
        grid[...,0]+=offsets[:,0]*(32/max(w-1,1))
        grid[...,1]+=offsets[:,1]*(32/max(h-1,1))
        grid[...,2]+=offsets[:,2]*(8/max(t-1,1))
        warped=F.grid_sample(parent_rgb.float(),grid,mode='bilinear',padding_mode='border',align_corners=True)
        return (warped+correction.float()).clamp(-1,1)

    def forward(self,parent,detail,parent_rgb,reference_parent=None,backend='torch'):
        return self.finish(parent_rgb,self.prepare_detail(parent,detail,reference_parent),backend)
