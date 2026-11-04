"""Three-NFE RDD: differentiable, adapter-free coarse->middle->fine reuse.

All three calls share one trainable backbone. Both boundary noise and cached
block updates remain differentiable. No 250-step cache assumptions are reused.
"""
import sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'2d'))
from sample_rdd_phased_dmd3 import build_diffusions
from diffusion.rdd_phased_dmd import fresh_transition
from diffusion.cross_scale_reuse_config import layer_masks


def lift(delta, grid):
    b,n,c=delta.shape;g=int(n**.5)
    if g*g!=n:raise ValueError('Square spatial source required')
    return F.interpolate(delta.transpose(1,2).reshape(b,c,g,g),size=(grid,grid),mode='nearest').flatten(2).transpose(1,2)


class CascadeDMD(nn.Module):
    def __init__(self, backbone, family, checkpoint_blocks=True):
        super().__init__();self.backbone=backbone;self.family=family
        self.masks=layer_masks(family,len(backbone.blocks))
        self.diffusions=build_diffusions((300,300,400),(32,16,8))[::-1]
        self.checkpoint_blocks=checkpoint_blocks

    def stage(self, x,t,y,stage,cache=None, *, capture_layers=(), hidden_out=None):
        # Optional TRAINING-ONLY post-block taps; no hooks, parameters or changes
        # to the normal inference path. Keep the student graph intact.
        capture_layers=set(capture_layers)
        if capture_layers and hidden_out is None:
            raise ValueError('capture_layers requires a hidden_out dictionary')
        if any(i<0 or i>=len(self.backbone.blocks) for i in capture_layers):
            raise ValueError('Hidden layer indices must be zero-based valid block indices')
        m=self.backbone;grid=x.shape[-1]//m.patch_size
        h=m.x_embedder(x);cos,sin=m.get_rope(grid,grid,x.device)
        condition=m.t_embedder(t)+m.y_embedder(y,False)
        capture=set(self.masks.get(stage+1,()))
        next_cache={}
        for i,block in enumerate(m.blocks):
            before=h
            if i in self.masks[stage]:
                if cache is None or i not in cache:raise RuntimeError('Missing previous-stage residual')
                h=h+lift(cache[i],grid).to(h.dtype)
            elif self.checkpoint_blocks and torch.is_grad_enabled():
                h=checkpoint(block,h,condition,cos,sin,use_reentrant=False)
            else:h=block(h,condition,cos,sin)
            if i in capture:next_cache[i]=h-before  # Never detach during training.
            if i in capture_layers:hidden_out[i]=h
        raw=m.unpatchify(m.final_layer(h,condition),grid,grid)
        return raw,next_cache

    def clean_at(self,x,y,stage,cache=None, *, capture_layers=(), hidden_out=None):
        diff=self.diffusions[stage];raw_holder={};next_cache={}
        def network(z,t,y):
            raw,captured=self.stage(z,t,y,stage,cache,capture_layers=capture_layers,hidden_out=hidden_out)
            raw_holder['raw']=raw;next_cache.update(captured)
            return raw
        local_t=torch.full((x.shape[0],),diff.num_timesteps-1,device=x.device,dtype=torch.long)
        clean=diff.p_mean_variance(network,x,local_t,clip_denoised=False,model_kwargs={'y':y})['pred_xstart'].float()
        return clean,raw_holder['raw'],next_cache

    def forward(self,noises,y,teacher_inputs=None,student_blend=1., *, hidden_layers=None):
        state=noises[0];cache=None;cleans=[];raws=[];inputs=[]
        if hidden_layers is not None and any(s not in (0,1,2) for s in hidden_layers):
            raise ValueError('Hidden stage indices must be 0, 1 or 2')
        hidden=[{},{},{}] if hidden_layers is not None else None
        for stage,resolution in enumerate((8,16,32)):
            if teacher_inputs is not None and stage>0 and student_blend<1:
                # Same noise coupling; blend only curriculum INPUTS, not target
                # losses or the reuse operation. Final training is fully on-policy.
                state=student_blend*state+(1-student_blend)*teacher_inputs[stage].detach()
            inputs.append(state)
            taps={} if hidden is None else dict(capture_layers=hidden_layers.get(stage,()),hidden_out=hidden[stage])
            clean,raw,cache=self.clean_at(state,y,stage,cache,**taps)
            cleans.append(clean);raws.append(raw)
            if stage<2:
                state=fresh_transition(clean,(16,32)[stage],self.diffusions[stage+1],noise=noises[stage+1])
        result={'cleans':cleans,'raws':raws,'inputs':inputs}
        if hidden is not None:result['hidden']=hidden
        return result


def coupled_noises(batch,device,generator):
    y=torch.randint(0,1000,(batch,),device=device,generator=generator)
    noises=[torch.randn(batch,4,r,r,device=device,generator=generator) for r in (8,16,32)]
    return noises,y
