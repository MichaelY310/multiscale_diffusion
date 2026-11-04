"""Auxiliary objectives are explicit components, not altered path targets."""
import torch
import numpy as np
from .registry import register
from .operators import block_average,resize,block_dct
from .tree import mean_flat

@register('regularizer','scale_equivariance')
class ScaleEquivariance:
    supports_reconstruction=True
    def __init__(self,factor=2,mode='mean',weight=1.):self.factor=factor;self.mode=mode;self.weight=weight
    def down(self,x):
        shape=x.shape
        flat=x.flatten(0,1) if x.ndim==5 else x
        if self.mode=='mean':out=block_average(flat,self.factor)
        elif self.mode=='bil':out=resize(flat,scale=1/self.factor,mode='bilinear',align_corners=True)
        else:raise ValueError('Unknown resampling mode')
        return out.unflatten(0,shape[:2]) if x.ndim==5 else out
    def __call__(self,decoder,latents,images,reconstruction=None):
        prediction=decoder(self.down(latents));target=self.down(images)
        return self.weight*(mean_flat((prediction-target)**2) if reconstruction is None else reconstruction(prediction,target))

@register('regularizer','random_scale_equivariance')
class RandomScaleEquivariance(ScaleEquivariance):
    def __init__(self,factors=(2,4),mode='mean',weight=1.):
        if not factors or any(int(x)!=x or x<1 for x in factors):raise ValueError('Scale factors must be positive integers')
        super().__init__(mode=mode,weight=weight);self.factors=factors
    def __call__(self,decoder,latents,images,reconstruction=None):
        self.factor=int(np.random.choice(self.factors))
        return super().__call__(decoder,latents,images,reconstruction)

@register('model','native_perceptual_pyramid')
def perceptual_pyramid(checkpoint=None,synthetic_fixture=False,**kwargs):
    """Unchanged author VGG pyramid; local VGG state replaces its downloader."""
    import torchvision
    from types import SimpleNamespace
    from . import native
    from .recipes import state_file
    if checkpoint is None and not synthetic_fixture:raise ValueError('Perceptual loss needs local VGG19 weights')
    def vgg19(**unused):
        model=torchvision.models.vgg19(weights=None)
        if checkpoint is not None:model.load_state_dict(state_file(checkpoint),strict=True)
        return model
    view=SimpleNamespace(models=SimpleNamespace(vgg19=vgg19,VGG19_Weights=torchvision.models.VGG19_Weights))
    env=native.extract('diffusability/src/training/perceptual_loss.py',['AntiAliasInterpolation2d','Vgg19','ImagePyramid','PerceptualPyramidLoss'],dict(np=np,torch=torch,F=torch.nn.functional,torchvision=view))
    return env['PerceptualPyramidLoss'](**kwargs).eval().requires_grad_(False)

def native_dct_utilities():
    import math,einops,torch_dct
    from types import SimpleNamespace
    from . import native
    return native.extract('diffusability/src/utils/training_utils.py',['cut_dct2d_high_freqs','reg_dc_dct2d_high_freqs','zigzag_indices'],dict(torch=torch,math=math,einops=einops,dct_2d=torch_dct.dct_2d,idct_2d=torch_dct.idct_2d,misc=SimpleNamespace(unsqueeze_left=lambda w,x:w.reshape((1,)*(x.ndim-w.ndim)+w.shape))))

@register('regularizer','dct_scale_equivariance')
class DCTScaleEquivariance:
    supports_reconstruction=True
    def __init__(self,max_cut_ratio=.5,block_size=None,zigzag=False,compression=8,weight=1.):
        if not 0<=max_cut_ratio<=1:raise ValueError('Invalid DCT cut ratio')
        self.ratio=max_cut_ratio;self.block=block_size;self.zigzag=zigzag;self.compression=compression;self.weight=weight;self.cut=native_dct_utilities()['cut_dct2d_high_freqs']
    def __call__(self,decoder,latents,images,reconstruction=None):
        import math
        block=self.block or math.gcd(*latents.shape[-2:]);ratio=np.random.rand()*self.ratio
        z=self.cut(latents,block_size=block,zigzag=self.zigzag,cut_ratio=ratio)
        x=self.cut(images,block_size=block*self.compression,zigzag=self.zigzag,cut_ratio=ratio)
        prediction=decoder(z)
        return self.weight*(mean_flat((prediction-x)**2) if reconstruction is None else reconstruction(prediction,x))

@register('regularizer','repa')
class REPA:
    def __init__(self,weight=1.):self.weight=weight
    def __call__(self,teacher_features,projected_features):
        if len(teacher_features)!=len(projected_features):raise ValueError('REPA feature-list mismatch')
        result=0.
        for target,pred in zip(teacher_features,projected_features):
            target=torch.nn.functional.normalize(target,dim=-1);pred=torch.nn.functional.normalize(pred,dim=-1)
            result=result-(target*pred).sum(-1).flatten(1).mean(1)
        return self.weight*result/len(teacher_features)

@register('regularizer','repa_native_dfm')
class NativeDFMREPA:
    """The released DFM scalar reduction, including its sample-loop order."""
    def __init__(self,weight=1.):self.weight=weight
    def __call__(self,teacher_features,projected_features):
        if len(teacher_features)!=len(projected_features) or not teacher_features:raise ValueError('REPA feature-list mismatch')
        batch=teacher_features[0].shape[0];loss=0.
        for target,pred in zip(teacher_features,projected_features):
            if target.shape!=pred.shape:raise ValueError('REPA feature shape mismatch')
            for tj,pj in zip(target,pred):
                pj=torch.nn.functional.normalize(pj,dim=-1);tj=torch.nn.functional.normalize(tj,dim=-1)
                values=-(tj*pj).sum(dim=-1)
                loss+=torch.mean(values,dim=list(range(1,values.ndim)))
        return self.weight*(loss/(len(teacher_features)*batch))
