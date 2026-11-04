"""Paired GPU attention benchmark. Synthetic tensors, not video quality or NFE.

Dense: ordinary Flash SDPA. VSA20: native FastVideo router + coarse/gate + sparse.
ROOT: joint local/shared ROOT + SAME current coarse/gate, ROOT projection charged.
Current pooled: same new kernel, current compact projected K/V (strong control).
Physical GPUs6/7 only. No distributed/whole-model speed claimed.
"""
import argparse
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend,sdpa_kernel
from root_local_attention_reference import bank_bias
from root_local_plan import AttentionPlan,KERNEL_CONFIGS,spatial_pool_projected


def stats(xs):
    return dict(median_ms=statistics.median(xs),min_ms=min(xs),max_ms=max(xs),samples_ms=xs)


def measured(fn,repeats=5,loops=3):
    for _ in range(2): fn()
    torch.cuda.synchronize()
    out=[]
    for _ in range(repeats):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(loops): fn()
        b.record();b.synchronize()
        out.append(a.elapsed_time(b)/loops)
    return stats(out)


def paired_reduction(records,baseline,candidate='root_local',resamples=2000):
    values=[]
    for group in sorted({r['group'] for r in records}):
        rows={r['variant']:r['ms'] for r in records if r['group']==group}
        values.append(100*(1-rows[candidate]/rows[baseline]))
    rng=random.Random(431)
    boot=sorted(statistics.mean(rng.choices(values,k=len(values))) for _ in range(resamples))
    return dict(mean_percent=statistics.mean(values),lower95_percent=boot[int(.025*resamples)],
                upper95_percent=boot[min(resamples-1,int(.975*resamples))],group_percent=values,
                warning='Small-sample bootstrap; requires more repeats for a stable speed claim')


def dry_run():
    rows=[]
    for shape,win in (((8,28,52),4),((16,48,80),8)):
        for divisor in (16,64):
            p=AttentionPlan(shape,win,math.prod(shape)//divisor)
            rows.append(dict(patch_grid=shape,tokens=p.tokens,local=p.local,root=p.root_slots,
                windows=p.tokens//p.local,vsa_blocks=p.blocks,keep=p.keep,actual_density=p.keep/p.blocks,
                bias=p.bias,window_map_valid=p.window_map().numel()==p.tokens))
    return dict(gpu_run=False,compiled=False,cases=rows,allowed_gpus=['6','7'],
        variants=['flash_dense','native_vsa20','root_local_with_projection_and_current_coarse','current_pooled_same_kernel'],
        scope='SP1 independent GPU repetitions, NOT SP2 or whole-model/video speed',
        bank_source='Synthetic BF16 feature bank; N/16 matches spatial421 ROOT count, N/64 is a compressed slot-budget ablation, not an implemented ROOT compression adapter')


def validate():
    from root_local_triton import root_local_attention
    torch.manual_seed(431)
    n,d,r=192,128,91
    vals=[torch.randn(1,2,n,d,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    rk,rv=[torch.randn(1,2,r,d,device='cuda',dtype=torch.bfloat16) for _ in range(2)]
    ids=torch.randperm(n,device='cuda')
    beta=-.7
    q,k,v=[x.float() for x in vals]
    expected=torch.empty_like(q)
    for i in range(4):
        ix=ids[i*48:(i+1)*48]
        kk=torch.cat((k[:,:,ix],rk.float()),dim=2)
        vv=torch.cat((v[:,:,ix],rv.float()),dim=2)
        logits=q[:,:,ix]@kk.transpose(-1,-2)/math.sqrt(d)
        logits[:,:,:,48:]+=beta
        expected[:,:,ix]=logits.softmax(-1)@vv
    checks=[]
    # Exercise Q padding and ROOT K tails in every candidate configuration.
    rk=rk.transpose(1,2).contiguous().transpose(1,2)
    rv=rv.transpose(1,2).contiguous().transpose(1,2)
    for cfg in KERNEL_CONFIGS:
        actual=root_local_attention(*vals,rk,rv,ids,48,beta,*cfg)
        delta=actual.float()-expected
        rel=float(delta.norm()/expected.norm())
        if rel>.009 or not torch.isfinite(actual).all():raise AssertionError(('GPU correctness',cfg,rel))
        checks.append(dict(config=cfg,relative_l2=rel,max_abs=float(delta.abs().max())))
    return dict(checks=checks,reference='GPU FP32 literal joint softmax',tail_test=True,permuted_map=True,noncontiguous_root_heads=True)


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('6','7'):
        raise ValueError('Exactly physical GPU6 or7 required')
    if a.repeats<3:raise ValueError('At least3 paired groups required')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError(out)
    torch.cuda.set_device(0);torch.set_num_threads(2)
    if torch.cuda.mem_get_info()[0]<25*2**30:raise RuntimeError('Need25GiB free, no eviction')
    torch.cuda.set_per_process_memory_fraction(.45)
    torch.backends.cuda.matmul.allow_tf32=False
    root=Path(a.root);fv=root/'third_party/FastVideo-ref'
    sys.path.insert(0,str(fv/'fastvideo-kernel/python'))
    from fastvideo_kernel.ops import fused_block_mean,video_sparse_attn
    from root_local_triton import root_local_attention
    report=dict(complete=False,physical_gpu=os.environ['CUDA_VISIBLE_DEVICES'],device=torch.cuda.get_device_name(),
                torch=torch.__version__,scope='Synthetic attention pipeline, SP1 independent GPU replicas; NOT model/NFE/video speed or trained quality',
                correctness=validate(),rows=[],seed=431,baseline='native FastVideo64 VSA20 including current coarse/gate and online router',
                dense='Flash SDPA only, no VSA extras (strong dense baseline)',
                bank_source='Independent synthetic BF16 ROOT-sized features, NOT captured pretrained ROOT. N/64 is a slot-budget ablation; actual ROOT compression not implemented',
                missing=['Actual model and root producer','ROOT layer normalization/modulation/physical RoPE adapter',
                         'ROOT broadcast/SP2','Training/backward/quality','VAE'],
                common_excluded='Current fine Q/K/V and gate projection, current physical RoPE; inputs already supplied',
                root_cost_included='Two dense BF16 K/V projections and contiguous head layout; actual ROOT compression, layer-specific norm/modulation/RoPE still missing')
    report['current_control_limit']='N/16 uses eager FP32 torch mean, not fastest fused pooling; N/64 reuses already-produced VSA summaries. A ROOT win over eager pooling alone cannot establish novelty.'
    for shape,win in (((8,28,52),4),((16,48,80),8)):
        if a.large_only and shape[0]!=16:continue
        t,h,w=shape;n=t*h*w;heads=12;d=128;hidden=heads*d
        torch.manual_seed(431+n)
        q,k,v,gate=[torch.randn(1,heads,n,d,device='cuda',dtype=torch.bfloat16) for _ in range(4)]
        plan=AttentionPlan(shape,win,n//16)
        tile=plan.cube_map().to('cuda')
        inv=torch.argsort(tile)
        window=plan.window_map().to('cuda')
        local=plan.local
        blocks=plan.blocks;keep=plan.keep
        sizes=torch.full((blocks,),64,device='cuda',dtype=torch.int32)
        def native_coarse(qt,kt,vt):
            qc,kc,vc=[fused_block_mean(x,sizes,64) for x in (qt,kt,vt)]
            score=qc@kc.transpose(-1,-2)/math.sqrt(d)
            prob=score.softmax(-1)
            oc=(prob@vc).unsqueeze(3).expand(-1,-1,-1,64,-1).reshape_as(qt)
            return score,oc,kc,vc
        def dense():
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):return F.scaled_dot_product_attention(q,k,v)
        def vsa():
            qt,kt,vt,gt=[x[:,:,tile].contiguous() for x in (q,k,v,gate)]
            return video_sparse_attn(qt,kt,vt,sizes,sizes,keep,
                block_size=(4,4,4),compress_attn_weight=gt)[:,:,inv]
        # Matched fixed oracle work: same original CURRENT64 coarse branch.
        def coarse_physical():
            qt,kt,vt=[x[:,:,tile].contiguous() for x in (q,k,v)]
            _,oc,kc,vc=native_coarse(qt,kt,vt)
            return oc[:,:,inv]*gate,kc,vc
        for r in (n//16,n//64):
            feature=torch.randn(1,r,hidden,device='cuda',dtype=torch.bfloat16)
            wk,wv=[torch.randn(hidden,hidden,device='cuda',dtype=torch.bfloat16)/math.sqrt(hidden) for _ in range(2)]
            def bank():
                return [(feature@weight).view(1,r,heads,d).transpose(1,2).contiguous() for weight in (wk,wv)]
            rk,rv=bank();beta=bank_bias(r,local,.2)
            tuning=[]
            for cfg in KERNEL_CONFIGS:
                fn=lambda:root_local_attention(q,k,v,rk,rv,window,local,beta,*cfg)
                tuning.append(dict(config=list(cfg),**measured(fn,3,2)))
            cfg=tuple(min(tuning,key=lambda row:row['median_ms'])['config'])
            torch.cuda.reset_peak_memory_stats()
            def candidate():
                ck,cv=bank()
                branch=root_local_attention(q,k,v,ck,cv,window,local,beta,*cfg)
                co,_,_=coarse_physical()
                return branch+co
            def current():
                co,kc,vc=coarse_physical()
                if r==n//64:ck,cv=kc,vc
                else:
                    # Strong current-feature counter: pool already projected K/V,
                    # so no redundant D x D projections forced on the control.
                    ck,cv=spatial_pool_projected(k,shape),spatial_pool_projected(v,shape)
                return root_local_attention(q,k,v,ck,cv,window,local,beta,*cfg)+co
            # Interleaved whole pipelines, not comparing unrelated warm runs.
            records=[]
            variants={'dense':dense,'vsa20':vsa,'root_local':candidate,'current_pooled':current}
            for f in variants.values():f()
            for group in range(a.repeats):
                names=list(variants)
                names=names[group%len(names):]+names[:group%len(names)]
                if group%2:names.reverse()
                for name in names:
                    row=measured(variants[name],1,2)
                    records.append(dict(group=group,variant=name,ms=row['median_ms']))
            times={name:stats([row['ms'] for row in records if row['variant']==name]) for name in variants}
            base=times['vsa20']['median_ms'];cand=times['root_local']['median_ms'];den=times['dense']['median_ms']
            report['rows'].append(dict(shape=shape,n=n,local=local,root=r,root_bias=beta,best_config=cfg,
                keep_blocks=keep,actual_vsa_density=keep/blocks,root_feature_mib=feature.numel()*2/2**20,
                times=times,records=records,tuning=tuning,
                root_vs_vsa_reduction_percent=100*(1-cand/base),root_vs_dense_reduction_percent=100*(1-cand/den),
                paired_vs_vsa=paired_reduction(records,'vsa20'),paired_vs_dense=paired_reduction(records,'dense'),
                paired_vs_current_pool=paired_reduction(records,'current_pooled'),
                measured_root_project=measured(bank,3,5),combined_variant_peak_gib=torch.cuda.max_memory_allocated()/2**30))
            (out/'progress.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report['rows'][-1]),flush=True)
        del q,k,v,gate,feature,wk,wv,rk,rv
        torch.cuda.empty_cache()
    report['complete']=True
    (out/'metrics.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output')
    p.add_argument('--repeats',type=int,default=5)
    p.add_argument('--large-only',action='store_true')
    p.add_argument('--dry-run',action='store_true',help='Validate plans on CPU only, without importing Triton or initializing CUDA')
    args=p.parse_args()
    if args.dry_run:print(json.dumps(dry_run(),indent=2))
    else:
        if not args.output:p.error('--output required unless --dry-run')
        with torch.inference_mode():main(args)
