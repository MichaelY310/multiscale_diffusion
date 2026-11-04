"""Inference-only fused parent gather + flow interpolation + RGB correction.

Written for future GPU validation; NO CUDA correctness/timing claim yet.
The same executor is available to the ordinary forecast/postfinal controls.
"""
import torch
try:
    import triton
    import triton.language as tl
except ImportError:
    triton=None


if triton is not None:
    @triton.jit
    def _render(P,F,C,O,N:tl.constexpr,T:tl.constexpr,H:tl.constexpr,W:tl.constexpr,
                PT:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
                FH:tl.constexpr,FW:tl.constexpr,BLOCK:tl.constexpr):
        i=tl.program_id(0)*BLOCK+tl.arange(0,BLOCK)
        valid=i<N
        x=i%W
        y=i//W%H
        t=i//(W*H)%T
        channel=i//(W*H*T)%3
        batch=i//(3*W*H*T)
        fx=x.to(tl.float32)*(FW-1)/max(W-1,1)
        fy=y.to(tl.float32)*(FH-1)/max(H-1,1)
        x0=tl.floor(fx).to(tl.int32)
        y0=tl.floor(fy).to(tl.int32)
        x1=tl.minimum(x0+1,FW-1)
        y1=tl.minimum(y0+1,FH-1)
        wx=fx-x0
        wy=fy-y0
        base=(batch*3*T+t)*FH*FW
        px=x.to(tl.float32)*(PW-1)/max(W-1,1)
        py=y.to(tl.float32)*(PH-1)/max(H-1,1)
        pt=t.to(tl.float32)*(PT-1)/max(T-1,1)
        for d in tl.static_range(3):
            offset=base+d*T*FH*FW
            f00=tl.load(F+offset+y0*FW+x0,valid,other=0).to(tl.float32)
            f01=tl.load(F+offset+y0*FW+x1,valid,other=0).to(tl.float32)
            f10=tl.load(F+offset+y1*FW+x0,valid,other=0).to(tl.float32)
            f11=tl.load(F+offset+y1*FW+x1,valid,other=0).to(tl.float32)
            v=(f00*(1-wx)+f01*wx)*(1-wy)+(f10*(1-wx)+f11*wx)*wy
            v=2/(1+tl.exp(-2*v))-1
            if d==0:
                px=px+v*16*(PW-1)/max(W-1,1)
            elif d==1:
                py=py+v*16*(PH-1)/max(H-1,1)
            else:
                pt=pt+v*4*(PT-1)/max(T-1,1)
        px=tl.minimum(tl.maximum(px,0),PW-1)
        py=tl.minimum(tl.maximum(py,0),PH-1)
        pt=tl.minimum(tl.maximum(pt,0),PT-1)
        xa=tl.floor(px).to(tl.int32)
        ya=tl.floor(py).to(tl.int32)
        ta=tl.floor(pt).to(tl.int32)
        wx=px-xa
        wy=py-ya
        wt=pt-ta
        value=tl.full((BLOCK,),0,tl.float32)
        parent_base=(batch*3+channel)*PT*PH*PW
        for corner in tl.static_range(8):
            cx=corner%2
            cy=corner//2%2
            ct=corner//4
            xi=tl.minimum(xa+cx,PW-1)
            yi=tl.minimum(ya+cy,PH-1)
            ti=tl.minimum(ta+ct,PT-1)
            weight=(wx if cx else 1-wx)*(wy if cy else 1-wy)*(wt if ct else 1-wt)
            p=tl.load(P+parent_base+(ti*PH+yi)*PW+xi,valid,other=0).to(tl.float32)
            value=value+weight*p
        correction=tl.load(C+i,valid,other=0).to(tl.float32)
        tl.store(O+i,tl.minimum(tl.maximum(value+correction,-1),1),valid)


def fused_render(parent_rgb,prepared):
    if triton is None:
        raise RuntimeError('Triton unavailable; use the PyTorch reference')
    correction,flow=prepared
    if parent_rgb.ndim==4:
        parent_rgb=parent_rgb.unsqueeze(0)
    if any(x.requires_grad for x in (parent_rgb,correction,flow)):
        raise ValueError('Inference-only kernel; train with PyTorch finish')
    if not all(x.is_cuda and x.device==correction.device for x in (parent_rgb,flow,correction)):
        raise ValueError('Matched CUDA device required')
    if correction.ndim!=5 or flow.ndim!=5 or parent_rgb.ndim!=5:
        raise ValueError('BCTHW tensors required')
    b,c,t,h,w=correction.shape
    if c!=3 or flow.shape[:3]!=(b,3,t) or parent_rgb.shape[:2]!=(b,3):
        raise ValueError('RGB/flow shape mismatch')
    parent_rgb,flow,correction=(x.contiguous() for x in (parent_rgb,flow,correction))
    output=torch.empty(correction.shape,device=correction.device,dtype=torch.float32)
    n=output.numel()
    _render[(triton.cdiv(n,256),)](parent_rgb,flow,correction,output,n,t,h,w,
        *parent_rgb.shape[2:],*flow.shape[3:],BLOCK=256,num_warps=4)
    return output
