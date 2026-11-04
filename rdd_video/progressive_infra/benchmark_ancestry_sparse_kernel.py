"""Matched exact sparse/coarse attention timing, not denoiser speed/quality."""
from pathlib import Path
import argparse
import json
import os
import statistics
import sys
import time
import math
import numpy as np
import torch

from ancestry_sparse_kernel import ancestry_sparse_attention
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sample_fastwan_native_vsa import add_fastvideo_paths
from attention_analysis.rdd_c2f_route_triton import lift_c2f_route


def main(a):
    root=Path(a.root)
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    torch.cuda.set_device(0)
    if torch.cuda.mem_get_info()[0]<12*2**30:
        raise RuntimeError('At least12GiB free required; never evict other jobs')
    torch.cuda.set_per_process_memory_fraction(.2)
    fv=root/'third_party/FastVideo-ref'
    add_fastvideo_paths(fv,fv/'fastvideo-kernel')
    from fastvideo_kernel.triton_kernels.block_sparse_attn_triton import triton_block_sparse_attn_forward
    from fastvideo_kernel.ops import fused_block_mean
    dev='cuda'
    hw=240
    ratio=4
    n=hw*ratio*64
    archive=np.load(root/'eval_results/progressive_infra_20261003/route_layout_audit1280/packed_masks.npz')
    mask=np.unpackbits(archive[f's0_l{a.layer}'],count=12*hw*hw).reshape(1,12,hw,hw)
    rows=np.stack([np.flatnonzero(row) for row in mask.reshape(-1,hw)]).reshape(1,12,hw,-1)
    index=torch.as_tensor(rows,dtype=torch.int32,device=dev)
    count=torch.full((1,12,hw),index.shape[-1],device=dev,dtype=torch.int32)
    key_sizes=torch.full((hw*ratio,),64,device=dev,dtype=torch.int32)
    generator=torch.Generator(device=dev).manual_seed(81711)
    q,k,v,gate=[torch.randn((1,12,n,128),device=dev,dtype=torch.bfloat16,generator=generator) for _ in range(4)]
    fine_index,fine_count=lift_c2f_route(index,count,(1,12,20),(4,12,20))
    cfg={'native':None,'native_cached':None,'adjacent64':(1,4,3),'adjacent64s5':(1,4,5),
         'adjacent64w8':(1,8,3),'adjacent64w8s5':(1,8,5),'siblings128w4':(2,4,3),
         'siblings128w8':(2,8,3),'siblings128s5':(2,8,5),'siblings256':(4,8,3)}
    def sparse(name):
        if name=='native_cached':
            return triton_block_sparse_attn_forward(q,k,v,fine_index,fine_count,key_sizes)[0]
        if name=='native':
            fi,fc=lift_c2f_route(index,count,(1,12,20),(4,12,20))
            return triton_block_sparse_attn_forward(q,k,v,fi,fc,key_sizes)[0]
        group,warps,stages=cfg[name]
        return ancestry_sparse_attention(q,k,v,index,count,key_sizes,hw=hw,ratio=ratio,
            siblings=group,warps=warps,stages=stages)
    def full(name):
        # ORIGINAL64-token coarse branch, kept independently of sibling CTA.
        qc,kc,vc=[fused_block_mean(x,key_sizes,64) for x in (q,k,v)]
        oc=((qc@kc.transpose(-2,-1)/math.sqrt(128)).softmax(-1)@vc).view(1,12,hw*ratio,1,128)
        oc=oc.repeat(1,1,1,64,1).view(1,12,n,128)
        return oc*gate+sparse(name)
    report=dict(complete=False,shape=list(q.shape),physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),
        route_source=f'actual FastWan root layer{a.layer}',keep=float(index.shape[-1]/hw),
        warning='Operator ONLY; no model speed or quality claim. Native includes current route lifting; native_cached lifts once before timing and is the conservative baseline. All full-attention comparisons retain current original64-token coarse/gate branch.',
        checks={},timings={})
    report['timing_scope']='sparse_only' if a.sparse_only else 'sparse_plus_current_coarse_gate'
    measured = sparse if a.sparse_only else full
    with torch.inference_mode():
        reference=sparse('native_cached')
        for name in cfg:
            for _ in range(2):
                result=measured(name)
            torch.cuda.synchronize()
            error=(sparse(name).float()-reference.float())
            rel=float(error.norm()/reference.float().norm())
            report['checks'][name]=dict(relative_l2=rel,max_abs=float(error.abs().max()))
            if rel>.01:
                raise AssertionError((name,rel))
            measurements=[]
            for g in range(4):
                order=['native_cached',name,name,'native_cached'] if g%2==0 else [name,'native_cached','native_cached',name]
                for current in order:
                    s,e=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(10):
                        measured(current)
                    e.record()
                    e.synchronize()
                    measurements.append(dict(group=g,variant=current,milliseconds=s.elapsed_time(e)/10))
            base=statistics.median(x['milliseconds'] for x in measurements if x['variant']=='native_cached')
            cand=statistics.median(x['milliseconds'] for x in measurements if x['variant']==name)
            report['timings'][name]=dict(native_ms=base,candidate_ms=cand,reduction_percent=100*(1-cand/base),records=measurements)
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(dict(variant=name,baseline_ms=base,candidate_ms=cand,rel_error=rel)),flush=True)
    report['complete']=True
    (out/'metrics.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True)
    p.add_argument('--layer',type=int,default=14)
    p.add_argument('--sparse-only',action='store_true')
    main(p.parse_args())
