"""Training-required mean/detail process for progressive rendering.

Not a new sampler for an ordinary pretrained Wan velocity model. Predictors
below return CLEAN band endpoints; conditional distillation must teach them.
No previous hidden state, residual, attention output or KV is reused.
"""
from dataclasses import dataclass
from typing import Callable
import math
import torch


def split_mean_detail(fine):
    if fine.ndim!=5 or fine.shape[2]%2:
        raise ValueError('Expected BCTHW with an even latent-time dimension')
    even,odd=fine[:,:,0::2],fine[:,:,1::2]
    return (even+odd)/2,(even-odd)/2


def merge_mean_detail(parent,detail):
    if parent.ndim!=5 or parent.shape!=detail.shape:
        raise ValueError('Matched BCTHW parent and detail required')
    return torch.stack((parent+detail,parent-detail),3).flatten(2,3)


def detail_noise_like(parent,generator):
    # Q=I-UP, P is pair averaging. Q epsilon is (+g,-g), Var(g)=1/2
    # for epsilon~N(0,I). Unit-variance g would incorrectly double Q noise.
    return torch.randn(parent.shape,device=parent.device,dtype=parent.dtype,
                       generator=generator)/math.sqrt(2)


def conditional_noisy_detail(clean_detail,sigma,generator):
    if not 0<=sigma<=1:
        raise ValueError('sigma must lie in [0,1]')
    return (1-sigma)*clean_detail+sigma*detail_noise_like(clean_detail,generator)


@dataclass(frozen=True)
class BandStage:
    level:int
    sigma:float
    clean:torch.Tensor


def sample_three_committed_stages(root_shape,predict_root:Callable,
        predict_detail:Callable,*,sigmas=(1.,.757,.522),seed=0,
        device='cpu',dtype=torch.float32,on_commit=None):
    """3 fresh clean-endpoint calls; preserves every committed parent.

    root_shape is B,C,T,H,W. Output has 4T frames in latent time.
    predict_root(noise,sigma) -> clean root.
    predict_detail(parent,noise,sigma,level) -> clean h of parent's shape.
    All sigmas are explicit data/noise values, not raw shifted-Wan t labels.
    callback receives committed clean tensors without detaching: joint training
    can backpropagate through every stage. It must NOT mutate them in-place.
    """
    if len(root_shape)!=5 or len(sigmas)!=3 or any(not 0<s<=1 for s in sigmas):
        raise ValueError('Positive BCTHW root and three sigma values required')
    if any(n<=0 for n in root_shape):
        raise ValueError('Empty root shape')
    generator=torch.Generator(device=device).manual_seed(seed)
    noise=torch.randn(root_shape,device=device,dtype=dtype,generator=generator)*sigmas[0]
    clean=predict_root(noise,sigmas[0])
    if clean.shape!=noise.shape:
        raise ValueError('Root predictor shape mismatch')
    commits=[BandStage(0,sigmas[0],clean)]
    if on_commit:
        on_commit(commits[-1])
    for level,sigma in enumerate(sigmas[1:],1):
        noisy_h=detail_noise_like(clean,generator)*sigma
        h=predict_detail(clean,noisy_h,sigma,level)
        clean=merge_mean_detail(clean,h)
        commits.append(BandStage(level,sigma,clean))
        if on_commit:
            on_commit(commits[-1])
    return clean,commits


def codec_distillation_loss(predicted_rgb,teacher_rgb,temporal_weight=.1):
    """Offline decoder distillation only, not a conditional diffusion score.

    Teacher RGB must be the decode of the SAME reconstructed full latent.
    No ordinary teacher queries with only detail coordinates corrupted.
    """
    if predicted_rgb.shape!=teacher_rgb.shape or predicted_rgb.ndim!=5:
        raise ValueError('Matched RGB BCTHW tensors required')
    pixel=(predicted_rgb-teacher_rgb).abs().mean()
    if predicted_rgb.shape[2]>1:
        temporal=(predicted_rgb.diff(dim=2)-teacher_rgb.diff(dim=2)).abs().mean()
    else:
        temporal=pixel*0
    return dict(total=pixel+temporal_weight*temporal,pixel=pixel,temporal=temporal)
