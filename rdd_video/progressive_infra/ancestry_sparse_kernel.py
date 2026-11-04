"""Exact shared-route execution for uniform temporal C2F descendants.

Fresh Q/K/V, independent row softmax, no output cache. Only sparse branch;
caller must retain the current 64-token coarse branch/gates separately.
This kernel is NOT universally progressive-exclusive: ordinary shared routing
can satisfy the same contract. Performance requires measured controls.
"""
import math
import types
import torch
import triton
import triton.language as tl


@triton.jit
def _uniform_lift(Index,Count,FineIndex,FineCount,
                  ParentN:tl.constexpr,FineN:tl.constexpr,HW:tl.constexpr,
                  Ratio:tl.constexpr,Keep:tl.constexpr,Block:tl.constexpr):
    row=tl.program_id(0)
    bh=row//FineN
    query=row%FineN
    parent=query//HW//Ratio*HW+query%HW
    count=tl.load(Count+bh*ParentN+parent)
    slots=tl.arange(0,Block)
    key=tl.load(Index+(bh*ParentN+parent)*Keep+slots//Ratio,
                mask=slots<count*Ratio,other=0)
    finekey=(key//HW*Ratio+slots%Ratio)*HW+key%HW
    tl.store(FineIndex+row*Keep*Ratio+slots,finekey,mask=slots<count*Ratio)
    tl.store(FineCount+row,count*Ratio)


def lift_uniform_routes(index,count,*,hw,ratio):
    """Vectorized exact uniform lift, avoiding huge static-loop compilation.

    Used by the NATIVE sparse control too: do not inflate its execution or
    cold-start overhead to create a sibling-kernel advantage.
    """
    b,h,parent_n,keep=index.shape
    if parent_n%hw or ratio not in (2,4):
        raise ValueError('Uniform temporal ratio2/4 required')
    index=index.contiguous()
    count=count.contiguous()
    fine_n=parent_n*ratio
    fi=torch.empty((b,h,fine_n,keep*ratio),device=index.device,dtype=torch.int32)
    fc=torch.empty((b,h,fine_n),device=index.device,dtype=torch.int32)
    _uniform_lift[(b*h*fine_n,)](index,count,fi,fc,ParentN=parent_n,
        FineN=fine_n,HW=hw,Ratio=ratio,Keep=keep,
        Block=triton.next_power_of_2(keep*ratio),num_warps=4)
    return fi,fc


def validate_uniform_lift(device):
    """Independent CPU reference, including padded rows and two time planes."""
    generator=torch.Generator().manual_seed(76123)
    records=[]
    for ratio in (2,4):
        hw,parent_n,keep=6,12,7
        index=torch.randint(parent_n,(1,2,parent_n,keep),generator=generator,dtype=torch.int32)
        count=torch.randint(1,keep+1,(1,2,parent_n),generator=generator,dtype=torch.int32)
        fi,fc=lift_uniform_routes(index.to(device),count.to(device),hw=hw,ratio=ratio)
        fi,fc=fi.cpu(),fc.cpu()
        for head in range(2):
            for query in range(parent_n*ratio):
                parent=(query//hw//ratio)*hw+query%hw
                expected=[]
                for key in index[0,head,parent,:count[0,head,parent]]:
                    expected.extend([(int(key)//hw*ratio+child)*hw+int(key)%hw for child in range(ratio)])
                assert int(fc[0,head,query])==len(expected)
                torch.testing.assert_close(fi[0,head,query,:len(expected)],torch.tensor(expected,dtype=torch.int32),rtol=0,atol=0)
        records.append(dict(ratio=ratio,exact_index_and_count=True))
    return records


@triton.jit
def _ancestry_fwd(Q,K,V,O,Indices,Counts,Sizes,
                 qz:tl.constexpr,qh:tl.constexpr,qt:tl.constexpr,qd:tl.constexpr,
                 kz:tl.constexpr,kh:tl.constexpr,kt:tl.constexpr,kd:tl.constexpr,
                 vz:tl.constexpr,vh:tl.constexpr,vt:tl.constexpr,vd:tl.constexpr,
                 oz:tl.constexpr,oh:tl.constexpr,ot:tl.constexpr,od:tl.constexpr,
                 H:tl.constexpr,D:tl.constexpr,ParentN:tl.constexpr,HW:tl.constexpr,
                 Ratio:tl.constexpr,Keep:tl.constexpr,Group:tl.constexpr):
    work=tl.program_id(0)
    bh=tl.program_id(1)
    b=bh//H
    h=bh%H
    groups:tl.constexpr=Ratio//Group
    parent=work//groups
    phase=work%groups*Group
    rows=tl.arange(0,Group*64)
    dims=tl.arange(0,D)
    # Load sibling cubes without a Q permutation buffer.
    block=((parent//HW)*Ratio+phase+rows//64)*HW+parent%HW
    token=block*64+rows%64
    q=tl.load(Q+b*qz+h*qh+token[:,None]*qt+dims[None,:]*qd)
    m=tl.full((Group*64,),-float('inf'),tl.float32)
    z=tl.zeros((Group*64,),tl.float32)
    acc=tl.zeros((Group*64,D),tl.float32)
    count=tl.load(Counts+bh*ParentN+parent)*Ratio
    kp=Indices+(bh*ParentN+parent)*Keep
    cols=tl.arange(0,64)
    scale:tl.constexpr=1.4426950408889634/math.sqrt(D)
    for slot in range(count):
        pk=tl.load(kp+slot//Ratio)
        kb=(pk//HW*Ratio+slot%Ratio)*HW+pk%HW
        valid=tl.load(Sizes+kb)
        tokens=kb*64+cols
        k=tl.load(K+b*kz+h*kh+dims[:,None]*kd+tokens[None,:]*kt)
        logits=tl.dot(q,k)*scale
        logits=tl.where(cols[None,:]<valid,logits,-float('inf'))
        newm=tl.maximum(m,tl.max(logits,1))
        p=tl.exp2(logits-newm[:,None])
        correction=tl.exp2(m-newm)
        z=z*correction+tl.sum(p,1)
        acc=acc*correction[:,None]
        v=tl.load(V+b*vz+h*vh+tokens[:,None]*vt+dims[None,:]*vd)
        acc=tl.dot(p.to(tl.bfloat16),v,acc)
        m=newm
    acc=acc/z[:,None]
    tl.store(O+b*oz+h*oh+token[:,None]*ot+dims[None,:]*od,acc.to(O.dtype.element_ty))


def ancestry_sparse_attention(q,k,v,parent_index,parent_count,key_sizes,*,hw,ratio,
                              siblings=2,warps=8,stages=3):
    if q.dtype!=torch.bfloat16 or k.dtype!=q.dtype or v.dtype!=q.dtype:
        raise ValueError('BF16 inference only')
    if q.shape!=k.shape or q.shape!=v.shape or q.ndim!=4:
        raise ValueError('Identical BHND tensor shapes required')
    if torch.is_grad_enabled() and any(x.requires_grad for x in (q,k,v)):
        raise ValueError('Inference-only')
    if siblings not in (1,2,4) or ratio%siblings:
        raise ValueError('Choose 1/2/4 uniform temporal descendants')
    b,h,n,d=q.shape
    if d!=128 or n%64 or parent_index.shape[:3]!=(b,h,n//64//ratio):
        raise ValueError('Head128, exact64-token cubes and correctly sized parent routes')
    if parent_index.dtype!=torch.int32 or parent_count.dtype!=torch.int32:
        raise ValueError('Compact int32 parent routes required')
    o=torch.empty_like(q)
    _ancestry_fwd[(n//64//siblings,b*h)](q,k,v,o,parent_index,parent_count,key_sizes,
        *q.stride(),*k.stride(),*v.stride(),*o.stride(),H=h,D=d,
        ParentN=parent_index.shape[2],HW=hw,Ratio=ratio,Keep=parent_index.shape[-1],
        Group=siblings,num_warps=warps,num_stages=stages)
    return o


class SharedRouteController:
    """Minimal SP1 C2F with native CURRENT coarse output on every layer.

    Root route is lifted through uniform temporal descendants. The native and
    sibling executors consume exactly the same mask, never cached outputs.
    """
    attention_mode='c2f_route'
    def __init__(self,model):
        self.stage=0
        self.executor='native'
        self.enabled=True
        self.routes={}
        self.originals=[]
        for i,block in enumerate(model.blocks):
            impl=block.attn1.attn_impl
            original=impl.forward
            self.originals.append((impl,original))
            def forward(this,q,k,v,gate,metadata,_layer=i,_original=original):
                if not self.enabled:
                    return _original(q,k,v,gate,metadata)
                from fastvideo_kernel.ops import fused_block_mean,fused_topk_mask
                from fastvideo_kernel.triton_kernels.index import map_to_index
                from fastvideo_kernel.block_sparse_attn import block_sparse_attn_from_indices
                q,k,v,gate=(x.transpose(1,2).contiguous() for x in (q,k,v,gate))
                sizes=metadata.variable_block_sizes
                grid=tuple(int(x) for x in metadata.num_tiles)
                qc,kc,vc=[fused_block_mean(x,sizes,64) for x in (q,k,v)]
                scores=qc@kc.transpose(-2,-1)/math.sqrt(q.shape[-1])
                coarse=(scores.softmax(-1)@vc).unsqueeze(3).repeat(1,1,1,64,1).reshape_as(q)
                if self.stage==0:
                    mask=fused_topk_mask(scores,max(1,math.ceil(.2*math.prod(grid))))
                    index,count=map_to_index(mask)
                    index=index[...,:max(1,math.ceil(.2*math.prod(grid)))].contiguous()
                    self.routes[_layer]=(index,count,grid)
                    sparse=block_sparse_attn_from_indices(q,k,v,index,count,sizes)[0]
                else:
                    if self.executor=='current_grouped':
                        # Strong ordinary-model counter: infer a shared router
                        # from CURRENT fine features, not an ancestral cache.
                        # The same geometry is valid at fixed resolution.
                        ratio=2**self.stage
                        if grid[0]%ratio:
                            raise ValueError('Uniform current temporal groups required')
                        parent_grid=(grid[0]//ratio,*grid[1:])
                        b,h,_,d=qc.shape
                        hw=grid[1]*grid[2]
                        grouped=[]
                        for x in (qc,kc):
                            grouped.append(x.reshape(b,h,parent_grid[0],ratio,hw,d)
                                .mean(dim=3).reshape(b,h,-1,d))
                        router_scores=grouped[0]@grouped[1].transpose(-2,-1)/math.sqrt(d)
                        mask=fused_topk_mask(router_scores,max(1,math.ceil(.2*math.prod(parent_grid))))
                        index,count=map_to_index(mask)
                        index=index[...,:max(1,math.ceil(.2*math.prod(parent_grid)))].contiguous()
                    else:
                        index,count,parent_grid=self.routes[_layer]
                    if grid[1:]!=parent_grid[1:] or grid[0]%parent_grid[0]:
                        raise ValueError('Only uniform temporal refinement supported')
                    ratio=grid[0]//parent_grid[0]
                    if self.executor=='native':
                        fi,fc=lift_uniform_routes(index,count,hw=grid[1]*grid[2],ratio=ratio)
                        sparse=block_sparse_attn_from_indices(q,k,v,fi,fc,sizes)[0]
                    else:
                        sparse=ancestry_sparse_attention(q,k,v,index,count,sizes,
                            hw=grid[1]*grid[2],ratio=ratio,siblings=min(2,ratio),warps=8,stages=5)
                return (coarse*gate+sparse).transpose(1,2).contiguous()
            impl.forward=types.MethodType(forward,impl)

    def begin_model_forward(self,stage,shape):
        self.stage=stage
        if stage==0:
            self.routes.clear()

    def restore(self):
        for impl,original in self.originals:
            impl.forward=original
        self.routes.clear()
