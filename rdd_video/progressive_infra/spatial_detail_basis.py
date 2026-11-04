"""CPU-verifiable 2x2 mean constraint and exact high-band FFN algebra.

This is NOT an exact acceleration of current RDD/Wan. Whole-model use must
train fixed per-layer anchors and scale-separated normalization/modulation.
Cached anchors must not be silently replaced by a previous noisy prediction.
"""
import torch
import torch.nn.functional as F


def signs(like):
    # Rows: top-left, top-right, bottom-left, bottom-right. Three zero-mean
    # orthogonal columns; S.T @ S = 4I. Mean-normalized, not orthonormal Haar.
    return like.new_tensor([[1,1,1],[-1,1,-1],[1,-1,-1],[-1,-1,1]])


def split_spatial_mean_detail(x):
    if x.ndim!=5 or x.shape[-2]%2 or x.shape[-1]%2:
        raise ValueError('Even-H/W BCTHW required')
    cells=torch.stack((x[...,0::2,0::2],x[...,0::2,1::2],
                       x[...,1::2,0::2],x[...,1::2,1::2]),-1)
    mean=cells.mean(-1)
    detail=torch.einsum('...p,pb->...b',cells,signs(x))/4
    return mean,detail


def merge_spatial_mean_detail(mean,detail):
    if mean.ndim!=5 or detail.shape!=(*mean.shape,3):
        raise ValueError('BCTHW mean and BCTHW3 detail required')
    cells=mean.unsqueeze(-1)+torch.einsum('...b,pb->...p',detail,signs(detail))
    b,c,t,h,w=mean.shape
    return cells.reshape(b,c,t,h,w,2,2).permute(0,1,2,3,5,4,6).reshape(b,c,t,2*h,2*w)


def sample_spatial_detail_noise(mean,generator):
    # For full iid unit Gaussian, each mean-normalized detail has Var=1/4.
    return torch.randn((*mean.shape,3),device=mean.device,dtype=mean.dtype,
                       generator=generator)/2


def cache_parent_preactivation(mean_hidden,w1,b1):
    return mean_hidden@w1+b1


def high_band_ffn(detail_hidden,parent_preactivation,w1,w2):
    """3-row expensive GEMMs, FOUR fresh nonlinear rows.

    detail_hidden: [...,3,D]; cached parent_pre: [...,G].
    w1[D,G], w2[G,D]. Return [...,3,D].
    Exact HIGH part of an ordinary4-child FFN if its input mean has exactly
    the cached preactivation. Does NOT return the ordinary fine FFN mean.
    """
    if detail_hidden.shape[-2]!=3 or detail_hidden.shape[-1]!=w1.shape[0]:
        raise ValueError('Three D-wide detail rows required')
    s=signs(detail_hidden)
    pre=parent_preactivation.unsqueeze(-2)+torch.einsum('pb,...bg->...pg',s,detail_hidden@w1)
    # Every child's nonlinear response uses current detail; no stale GELU cache.
    current=F.gelu(pre)
    high_features=torch.einsum('pb,...pg->...bg',s,current)/4
    return high_features@w2


def dense_high_band_ffn_reference(mean_hidden,detail_hidden,w1,b1,w2,b2):
    s=signs(detail_hidden)
    children=mean_hidden.unsqueeze(-2)+torch.einsum('pb,...bd->...pd',s,detail_hidden)
    output=F.gelu(children@w1+b1)@w2+b2
    return torch.einsum('pb,...pd->...bd',s,output)/4
