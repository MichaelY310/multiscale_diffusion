"""CPU-only oracle for a NEW, training-required fine attention graph.

All Q/K/V are current except a root-generating feature bank. This is not an
equivalent VSA replacement or a complete Wan implementation. Caller supplies
K/Q already rotated ONCE in a consistent physical RoPE coordinate frame.
No GPU path, encoder, quantized GEMM or distributed runtime is implemented.
"""
from dataclasses import dataclass
import math
import torch


def spatial_tiles(shape, height=8, width=8):
    """Own full time axis, tile space only; return physical flattened indices."""
    t,h,w=shape
    if any(not isinstance(x,int) or x<1 for x in (t,h,w,height,width)):
        raise ValueError('Positive patch-grid dimensions required')
    grid=torch.arange(t*h*w).reshape(t,h,w)
    return tuple(grid[:,y:y+height,x:x+width].reshape(-1)
                 for y in range(0,h,height) for x in range(0,w,width))


def bank_bias(root_count, local_count, prior=.2):
    """Initial ROOT mass under equal logits, BEFORE learning, after /sqrt(d)."""
    if root_count<1 or local_count<1 or not 0<prior<1:
        raise ValueError('Invalid bank prior/counts')
    return math.log(prior/(1-prior))+math.log(local_count/root_count)


def _validate(q,k,v,rk,rv,tiles):
    if any(x.device.type!='cpu' for x in (q,k,v,rk,rv)):
        raise ValueError('This is a CPU-only reference')
    if any(x.ndim!=4 for x in (q,k,v,rk,rv)):
        raise ValueError('Expected [B,H,N,D] tensors')
    if q.shape!=k.shape or q.shape[:3]!=v.shape[:3] or rk.shape[:2]!=q.shape[:2]:
        raise ValueError('Current Q/K/V or ROOT batch/head mismatch')
    if rk.shape[:3]!=rv.shape[:3] or rk.shape[-1]!=q.shape[-1] or rv.shape[-1]!=v.shape[-1]:
        raise ValueError('ROOT K/V dimensions mismatch')
    if min(q.shape)<1 or min(rk.shape)<1 or v.shape[-1]<1:
        raise ValueError('Nonempty Q/ROOT bank required')
    if any(x.dtype!=q.dtype or x.dtype not in (torch.float32,torch.float64) for x in (q,k,v,rk,rv)):
        raise ValueError('Reference requires common FP32/FP64 dtype')
    if any(not torch.isfinite(x).all() for x in (q,k,v,rk,rv)):
        raise ValueError('Nonfinite input')
    if not isinstance(tiles,(tuple,list)) or not tiles or any(
        ids.device.type!='cpu' or ids.dtype!=torch.int64 or ids.ndim!=1 or ids.numel()==0 for ids in tiles):
        raise ValueError('Nonempty one-dimensional CPU int64 tiles required')
    indices=torch.cat(tiles)
    if indices.dtype!=torch.int64 or not torch.equal(torch.sort(indices).values,torch.arange(q.shape[2])):
        raise ValueError('Tiles must partition every physical query exactly once')


def _inputs(q,k,v,rk,rv,tiles,bias,valid,root_valid):
    _validate(q,k,v,rk,rv,tiles)
    b,h,n,_=q.shape
    if valid is None: valid=torch.ones(b,n,dtype=torch.bool)
    if root_valid is None: root_valid=torch.ones(b,rk.shape[2],dtype=torch.bool)
    for mask,shape in ((valid,(b,n)),(root_valid,(b,rk.shape[2]))):
        if mask.device.type!='cpu' or mask.dtype!=torch.bool or tuple(mask.shape)!=shape:
            raise ValueError('Validity mask shape/dtype/device mismatch')
    bias=torch.as_tensor(bias,dtype=q.dtype)
    if bias.device.type!='cpu' or not torch.isfinite(bias).all() or bias.numel() not in (1,h):
        raise ValueError('Bias must be finite scalar or per-head CPU tensor')
    return bias.reshape(1,-1,1,1),valid,root_valid


def _finish(outputs,tiles,coarse,gate,valid):
    y=torch.cat(outputs,dim=2)
    order=torch.argsort(torch.cat(tiles))
    y=y[:,:,order]
    if (coarse is None)!=(gate is None):
        raise ValueError('Coarse output and original gate must be supplied together')
    if coarse is not None:
        if coarse.shape!=y.shape or gate.shape!=y.shape or coarse.device.type!='cpu' or gate.device.type!='cpu':
            raise ValueError('Coarse output/gate shape or device mismatch')
        if coarse.dtype!=y.dtype or gate.dtype!=y.dtype or not torch.isfinite(coarse).all() or not torch.isfinite(gate).all():
            raise ValueError('Coarse output/gate dtype or finiteness mismatch')
        # Native VSA combination is ADDITIVE, not a convex gate blend.
        y=y+coarse*gate
    return y*valid[:,None,:,None]


def joint_dense(q,k,v,rk,rv,tiles,bias=0.,valid=None,root_valid=None,coarse=None,gate=None):
    """Literal concatenated-KV oracle; may materialize full score matrices."""
    bias,valid,root_valid=_inputs(q,k,v,rk,rv,tiles,bias,valid,root_valid)
    outputs=[]
    for ids in tiles:
        logits=torch.cat((q[:,:,ids]@k[:,:,ids].transpose(-1,-2)/math.sqrt(q.shape[-1]),
                          q[:,:,ids]@rk.transpose(-1,-2)/math.sqrt(q.shape[-1])+bias),dim=-1)
        allowed=torch.cat((valid[:,ids],root_valid),dim=-1)[:,None,None,:]
        logits=logits.masked_fill(~allowed,float('-inf'))
        m=logits.amax(dim=-1,keepdim=True)
        p=torch.exp(logits-torch.where(torch.isfinite(m),m,torch.zeros_like(m)))
        p=p/p.sum(dim=-1,keepdim=True).clamp_min(torch.finfo(q.dtype).tiny)
        out=p@torch.cat((v[:,:,ids],rv),dim=2)
        outputs.append(out*valid[:,None,ids,None])
    return _finish(outputs,tiles,coarse,gate,valid)


def joint_streaming(q,k,v,rk,rv,tiles,bias=0.,valid=None,root_valid=None,
                    coarse=None,gate=None,chunk=64):
    """Online joint softmax: shared ROOT bank, no bank clone per spatial tile.

    Python/tensor CPU correctness oracle, NOT a fused GPU kernel benchmark.
    Each score chunk is [B,H,tile_queries,chunk_keys], not full N×(R+L).
    Autograd retains chunks: this is NOT memory-efficient training backward.
    A production training kernel needs a recomputing/fused backward.
    Training gradients flow to ROOT producer; do not detach here.
    """
    if not isinstance(chunk,int) or chunk<1:
        raise ValueError('Positive integer chunk required')
    bias,valid,root_valid=_inputs(q,k,v,rk,rv,tiles,bias,valid,root_valid)
    outputs=[]
    for ids in tiles:
        qt=q[:,:,ids]
        m=torch.full((*qt.shape[:-1],1),float('-inf'),dtype=q.dtype)
        total=torch.zeros_like(m)
        acc=torch.zeros((*qt.shape[:-1],v.shape[-1]),dtype=q.dtype)
        for kb,vb,mask,offset in ((k[:,:,ids],v[:,:,ids],valid[:,ids],0.),(rk,rv,root_valid,bias)):
            for start in range(0,kb.shape[2],chunk):
                score=qt@kb[:,:,start:start+chunk].transpose(-1,-2)/math.sqrt(q.shape[-1])+offset
                score=score.masked_fill(~mask[:,None,None,start:start+chunk],float('-inf'))
                m_new=torch.maximum(m,score.amax(dim=-1,keepdim=True))
                safe=torch.where(torch.isfinite(m_new),m_new,torch.zeros_like(m_new))
                scale=torch.exp(m-safe)
                p=torch.exp(score-safe)
                total=total*scale+p.sum(dim=-1,keepdim=True)
                acc=acc*scale+p@vb[:,:,start:start+chunk]
                m=m_new
        outputs.append((acc/total.clamp_min(torch.finfo(q.dtype).tiny))*valid[:,None,ids,None])
    return _finish(outputs,tiles,coarse,gate,valid)


def layer_bank(root_feature,wk,wv):
    """Charged per-layer projection of ONE emitted root feature [B,R,D].

    Caller handles layer/time normalization and physical RoPE. Initialization
    may use existing fine Wk/Wv; this alone is NOT a trained memory consumer.
    """
    if root_feature.device.type!='cpu' or wk.device.type!='cpu' or wv.device.type!='cpu':
        raise ValueError('CPU-only bank reference')
    if root_feature.ndim!=3 or wk.ndim!=2 or wv.ndim!=2 or min(root_feature.shape)<1:
        raise ValueError('ROOT feature [B,R,D] and projection matrices required')
    if wk.shape[0]!=root_feature.shape[-1] or wv.shape[0]!=root_feature.shape[-1] or min(wk.shape[1],wv.shape[1])<1:
        raise ValueError('Projection dimensions mismatch')
    if any(x.dtype!=root_feature.dtype or x.dtype not in (torch.float32,torch.float64)
           or not torch.isfinite(x).all() for x in (root_feature,wk,wv)):
        raise ValueError('Finite common FP32/FP64 tensors required')
    return root_feature@wk,root_feature@wv


@dataclass(frozen=True)
class Budget:
    tokens: int=61440
    hidden: int=1536
    ffn: int=8960
    layers: int=30
    heads: int=12
    text_tokens: int=512
    fine_ranks: int=4
    density: float=.2
    local: int=1024
    root_slots: int=3840

    def calculate(self):
        for name in ('tokens','hidden','ffn','layers','heads','text_tokens','fine_ranks','local','root_slots'):
            value=getattr(self,name)
            if not isinstance(value,int) or value<1:
                raise ValueError('Positive integer budget dimensions required')
        if self.tokens%16 or self.hidden%self.heads or max(self.local,self.root_slots)>self.tokens:
            raise ValueError('Invalid spatial421/head geometry')
        if not math.isfinite(self.density) or not 0<self.density<=1:
            raise ValueError('Invalid VSA density')
        n,d,g=self.tokens,self.hidden,self.ffn
        ns=(n//16,n//4,n)
        # Self QKV/out (8ND²), VSA's D->D coarse gate (2ND²),
        # FFN (4NDG), text Q/out (4ND²), text K/V (4TD²).
        # The gate is retained in C and must not disappear from the budget.
        linear=lambda x:10*x*d*d+4*x*d*g+4*x*d*d+4*self.text_tokens*d*d
        text=lambda x:4*x*self.text_tokens*d
        attention=lambda x:4*self.density*x*x*d
        base_per_layer=sum(linear(x)+text(x)+attention(x) for x in ns)
        new_fine=4*n*(self.root_slots+self.local)*d
        # All fine ranks redundantly project the replicated ROOT feature.
        bank_project=4*self.fine_ranks*self.root_slots*d*d
        saved=attention(n)-new_fine-bank_project
        return dict(scope='Arithmetic illustration only; not latency/quality. Spatial421, fine-only replacement.',
                    self_ffn_text_matmul_reduction_percent=100*saved/base_per_layer,
                    fine_sparse_pair_fraction_of_vsa20=(self.root_slots+self.local)/(self.density*n),
                    root_feature_mib=self.root_slots*d*2/2**20,
                    layer_kv_scratch_mib=2*self.root_slots*d*2/2**20,
                    all_layer_kv_gib_reference=self.layers*2*self.root_slots*d*2/2**30,
                    root_project_tflop_aggregate=bank_project*self.layers/1e12,
                    baseline_matmul_tflop=base_per_layer*self.layers/1e12,
                    copied_feature_total_mib=(self.fine_ranks-1)*self.root_slots*d*2/2**20,
                    ignored_costs=['Norm/modulation/RoPE/pooling/gate activation/quantization',
                                   'Cache IO/broadcast/sync/actual kernel utilization',
                                   'Current coarse-output branch and routing, VAE',
                                   'Redundant replicated text projections may depend on SP backend'])
