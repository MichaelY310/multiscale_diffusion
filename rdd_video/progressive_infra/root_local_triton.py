"""Inference-only joint local + shared ROOT attention, no cached output.

Caller supplies Q/K already rotated once in a common physical coordinate frame.
Map partitions the physical sequence into equal-sized full-time spatial windows.
No ROOT bank is replicated per window, no N x (L+R) scores are materialized.
This changes the attention graph and requires training for quality claims.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _joint(Q, K, V, RK, RV, Map, O,
           qz:tl.constexpr,qh:tl.constexpr,qt:tl.constexpr,
           kz:tl.constexpr,kh:tl.constexpr,kt:tl.constexpr,
           vz:tl.constexpr,vh:tl.constexpr,vt:tl.constexpr,
           rz:tl.constexpr,rh:tl.constexpr,rt:tl.constexpr,
           sz:tl.constexpr,sh:tl.constexpr,st:tl.constexpr,
           H:tl.constexpr,N:tl.constexpr,D:tl.constexpr,L:tl.constexpr,
           R:tl.constexpr,Bias:tl.constexpr,M:tl.constexpr,C:tl.constexpr):
    work=tl.program_id(0)
    bh=tl.program_id(1).to(tl.int64)
    window=work//tl.cdiv(L,M)
    first=work%tl.cdiv(L,M)*M
    rows=first+tl.arange(0,M)
    cols=tl.arange(0,C)
    dims=tl.arange(0,D)
    ids=tl.load(Map+window*L+rows,rows<L,other=0).to(tl.int64)
    b=bh//H
    h=bh%H
    q=tl.load(Q+b*qz+h*qh+ids[:,None]*qt+dims[None,:],rows[:,None]<L,other=0)
    mx=tl.full((M,),-float('inf'),tl.float32)
    den=tl.zeros((M,),tl.float32)
    acc=tl.zeros((M,D),tl.float32)
    scale:tl.constexpr=1.4426950408889634/math.sqrt(D)
    for start in range(tl.cdiv(L,C)):
        j=start*C+cols
        ki=tl.load(Map+window*L+j,j<L,other=0).to(tl.int64)
        k=tl.load(K+b*kz+h*kh+ki[None,:]*kt+dims[:,None],j[None,:]<L,other=0)
        score=tl.dot(q,k)*scale
        score=tl.where(j[None,:]<L,score,-float('inf'))
        new=tl.maximum(mx,tl.max(score,1))
        p=tl.exp2(score-new[:,None])
        corr=tl.exp2(mx-new)
        den=den*corr+tl.sum(p,1)
        acc=acc*corr[:,None]
        v=tl.load(V+b*vz+h*vh+ki[:,None]*vt+dims[None,:],j[:,None]<L,other=0)
        acc=tl.dot(p.to(q.dtype),v,acc)
        mx=new
    for start in range(tl.cdiv(R,C)):
        j=start*C+cols
        k=tl.load(RK+b*rz+h*rh+j[None,:]*rt+dims[:,None],j[None,:]<R,other=0)
        score=tl.dot(q,k)*scale+Bias*1.4426950408889634
        score=tl.where(j[None,:]<R,score,-float('inf'))
        new=tl.maximum(mx,tl.max(score,1))
        p=tl.exp2(score-new[:,None])
        corr=tl.exp2(mx-new)
        den=den*corr+tl.sum(p,1)
        acc=acc*corr[:,None]
        v=tl.load(RV+b*sz+h*sh+j[:,None]*st+dims[None,:],j[:,None]<R,other=0)
        acc=tl.dot(p.to(q.dtype),v,acc)
        mx=new
    tl.store(O+(bh*N+ids[:,None])*D+dims[None,:],(acc/den[:,None]).to(O.dtype.element_ty),rows[:,None]<L)


def root_local_attention(q,k,v,rk,rv,window_map,local,bias=0.,qblock=64,kblock=64,warps=4,stages=3):
    """[B,H,N,D] -> [B,H,N,D]; common BF16/FP16, D64/128, inference only.

    window_map int64/int32 [N] must be a bijection in window-major order.
    Building/validating map belongs to cached shape metadata, not every layer.
    """
    if any(x.ndim!=4 for x in (q,k,v,rk,rv)) or not isinstance(local,int) or local<1:
        raise ValueError('BHND tensors and positive integer window required')
    b,h,n,d=q.shape
    if min(q.shape)<1 or q.shape!=k.shape or q.shape!=v.shape or d not in (64,128) or n%local:
        raise ValueError('Equal QKV, head64/128, equal windows dividing N required')
    if any(x.device!=q.device or x.dtype!=q.dtype or x.stride(-1)!=1 for x in (k,v,rk,rv)):
        raise ValueError('Common device/dtype and contiguous channel dimension required')
    if q.device.type!='cuda' or q.dtype not in (torch.bfloat16,torch.float16) or q.stride(-1)!=1:
        raise ValueError('CUDA BF16/FP16 only')
    if rk.shape!=rv.shape or rk.shape[:2]!=(b,h) or rk.shape[-1]!=d or rk.shape[2]<1:
        raise ValueError('ROOT shape mismatch/empty bank')
    if window_map.shape!=(n,) or window_map.device!=q.device or window_map.dtype not in (torch.int32,torch.int64) or not window_map.is_contiguous():
        raise ValueError('Contiguous CUDA integer window map [N] required')
    if not math.isfinite(bias) or abs(bias)>2e38 or qblock not in (32,64,128) or kblock not in (32,64,128) or warps not in (4,8) or stages not in (1,2,3,4,5):
        raise ValueError('Invalid kernel configuration')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v,rk,rv)):
        raise ValueError('Inference only; backward not implemented')
    out=torch.empty(q.shape,dtype=q.dtype,device=q.device)
    _joint[(n//local*triton.cdiv(local,qblock),b*h)](q,k,v,rk,rv,window_map,out,
        *q.stride()[:3],*k.stride()[:3],*v.stride()[:3],*rk.stride()[:3],*rv.stride()[:3],
        H=h,N=n,D=d,L=local,R=rk.shape[2],Bias=float(bias),M=qblock,C=kblock,
        num_warps=warps,num_stages=stages)
    return out
