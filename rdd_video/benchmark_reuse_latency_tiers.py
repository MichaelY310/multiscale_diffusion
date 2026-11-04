"""Aggressive direct/cascaded residual reuse. Matched REAL VAE total latency.

One identical resident backbone/VSA20/VAE in all arms. ABBA complete trajectories;
coarse output and both gates remain intact in every executed native block.
"""
import argparse
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from sample_fastwan_native_vsa import add_fastvideo_paths, sample_three_call_dmd
from benchmark_csra_prefix_feasibility import aggregate, group_bootstrap_ci

def spread(n):
    return sorted(set(round(1+i*27/(n-1)) for i in range(n)))


OLD_CONFIGS = {
    'baseline': {},
    'fine18': {2: spread(18)},
    'fine24': {2: spread(24)},
    'cascade24': {1: [i for i in range(30) if i not in (0,5,11,17,23,29)],
                  2: [i for i in range(30) if i not in (0,6,12,18,24,29)]},
    'cascade28': {1: list(range(1,29)), 2: list(range(1,29))},
}

# Human-readable masks: layer indices are zero-based.
def config(layers, middle=None, scale=1.):
    masks={2:list(layers)}
    if middle is not None: masks[1]=list(middle)
    return dict(masks=masks,scale=scale)

CONFIGS = {
    'baseline': dict(masks={},scale=1.),
    'fine18': config(spread(18)),
    'fine24': config(spread(24)),
    'fine28': config(range(1,29)),
    'fine29': config(range(29)),
    'prefix18': config(range(18)),
    'prefix24': config(range(24)),
    'prefix24_s075': config(range(24),scale=.75),
    'fine24_s075': config(spread(24),scale=.75),
    'fine24_s050': config(spread(24),scale=.5),
    'fine28_s075': config(range(1,29),scale=.75),
    'cascade12_24': config(spread(24),middle=spread(12)),
    'cascade18_24': config(spread(24),middle=spread(18)),
    'cascade24': dict(masks=OLD_CONFIGS['cascade24'],scale=1.),
    'cascade28': dict(masks=OLD_CONFIGS['cascade28'],scale=1.),
    'cascade_prefix18_24': config(range(24),middle=range(18)),
    'cascade_prefix24': config(range(24),middle=range(24)),
}

def main(a):
    root,out=Path(a.root),Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists(): return
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in getattr(a,'allowed_gpus',('0','1','2','3')): raise ValueError('Require one authorized GPU')
    dev=torch.device('cuda:0');torch.cuda.set_device(dev)
    if torch.cuda.mem_get_info()[0] < 30*2**30: raise RuntimeError('Need30GiB free; never evict others')
    torch.cuda.set_per_process_memory_fraction(.30,dev)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    fv=root/'third_party/FastVideo-ref';add_fastvideo_paths(fv,fv/'fastvideo-kernel')
    from attention_analysis.fastwan_native_attention import install_projected_cache_class_patch, FastWanNativeAttentionController
    from attention_analysis.one_call_cascade_controller import OneCallCascadeController
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.dits.wanvideo import WanTransformerBlock_VSA
    from fastvideo.models.loader.component_loader import TransformerLoader
    from wan.modules.t5 import T5EncoderModel
    from wan.modules.vae import WanVAE
    from sample_rdd_wan import save_video
    wan=root/'pretrained_models/Wan2.1-T2V-1.3B'
    text=T5EncoderModel(text_len=512,dtype=torch.float32,device=torch.device('cpu'),
        checkpoint_path=str(wan/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(wan/'google/umt5-xxl'))
    text.model.to(dev)
    with torch.inference_mode():
        tokens=text([a.prompt],dev)[0]
        context=torch.zeros((1,512,tokens.shape[-1]),device=dev,dtype=torch.bfloat16)
        context[0,:len(tokens)]=tokens.to(torch.bfloat16)
    text.model.cpu();del text,tokens;torch.cuda.empty_cache()
    install_projected_cache_class_patch(WanTransformerBlock_VSA)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(a.port),RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    maybe_init_distributed_environment_and_model_parallel(1,1)
    weights=root/'pretrained_models/FastWan2.1-T2V-1.3B-Diffusers/transformer'
    fa=FastVideoArgs(model_path=str(weights),num_gpus=1,dit_cpu_offload=False,dit_layerwise_offload=False,
        pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(),dit_precision='bf16'))
    fa.device=dev
    model=TransformerLoader().load(str(weights),fa).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    vae=WanVAE(z_dim=16,vae_pth=str(wan/'Wan2.1_VAE.pth'),dtype={'fp32':torch.float32,'bf16':torch.bfloat16}[a.vae_precision],device=dev)
    latent_t=(a.frames-1)//4+1
    if a.height % 16 or a.width % 16 or (a.frames-1)%4:
        raise ValueError('Output H/W must divide16; frames must be4n+1')
    shape=(1,16,latent_t,a.height//8,a.width//8)
    factors={'spatial421':[(4,1),(2,1),(1,1)], 'temporal421':[(1,4),(1,2),(1,1)],
             'temporal221':[(1,2),(1,2),(1,1)], 'st421':[(4,4),(2,2),(1,1)],
             'spatial221':[(2,1),(2,1),(1,1)], 'st221':[(2,2),(2,2),(1,1)]}[a.factors]
    if any(latent_t%factor_t for _,factor_t in factors): raise ValueError('Temporal ratio must divide latent frames')
    variants=['baseline']+a.variants.split(',')
    for v in variants:
        if v not in CONFIGS: raise ValueError(v)
        # Only actual scale changes may consume residuals. Do not relabel a
        # same-resolution timestep cache as an RDD-specific technique.
        for stage in CONFIGS[v]['masks']:
            if factors[stage-1] == factors[stage]:
                raise ValueError(f'{v}: stage {stage} does not change resolution')
    if any(v not in CONFIGS or v=='baseline' for v in variants[1:]): raise ValueError('Invalid variants')
    if len(variants)!=len(set(variants)): raise ValueError('Duplicate variants')
    def sample(controller=None):
        with contextlib.redirect_stdout(io.StringIO()),torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return sample_three_call_dmd(model,context,shape,factors,[1000,757,522],a.seed,.8,
                'noise_filling',dev,attention_controller=controller,boundary_executor='native',return_breakdown=True)
    ref=sample()[0]
    control=OneCallCascadeController(model)
    control.configure('vsa')
    no_op=sample(control)[0]
    torch.testing.assert_close(ref,no_op,atol=0,rtol=0)
    del ref,no_op
    cache_parity={}
    for name in variants[1:2]:
        masks=CONFIGS[name]['masks']
        control.lifetime_optimized=False
        control.configure('vsa',cross_stage_reuse_layers_by_stage=masks,cross_stage_reuse_stages=list(masks),cross_stage_reuse_scale=CONFIGS[name]['scale'])
        old=sample(control)[0]
        control.lifetime_optimized=True
        control.configure('vsa',cross_stage_reuse_layers_by_stage=masks,cross_stage_reuse_stages=list(masks),cross_stage_reuse_scale=CONFIGS[name]['scale'])
        new=sample(control)[0]
        torch.testing.assert_close(old,new,atol=0,rtol=0)
        if control.cross_stage_residual_cache: raise AssertionError('Unconsumed cache')
        cache_parity[name]=0.
        del old,new
    torch.cuda.empty_cache()
    protocol=dict(complete=False,version='direct_block_residual_no_adapter_v1',size=f'{a.width}x{a.height}',
        model=str(weights),shape=shape,factors=factors,frames=a.frames,fps=16,prompt=a.prompt,seed=a.seed,
        steps=[1000,757,522],NFE=3,boundary='noise_filling',sparse_keep=.20,
        layers_by_stage_zero_based={v:CONFIGS[v]['masks'] for v in variants},
        reuse_scale={v:CONFIGS[v]['scale'] for v in variants},config_family=a.factors,
        source='immediately previous stage delta; cascaded if previous block was also reused',
        scale=1.,new_parameters=0,offload=False,TF32=False,
        vae=f'{a.vae_precision} identical real decode after every trajectory',baseline_noop_max_abs=0.,
        cache_lifetime_parity_max_abs=cache_parity,
        warning='FastWan is NOT RDD-trained. No new video finetuning. Shared GPU: exploratory latency only.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),entries=[])
    latents={}
    def run(variant,export=False):
        masks=CONFIGS[variant]['masks']
        control.configure('vsa',cross_stage_reuse_layers_by_stage=masks,cross_stage_reuse_stages=list(masks),cross_stage_reuse_scale=CONFIGS[variant]['scale'])
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        result=sample(control);torch.cuda.synchronize();denoise=time.perf_counter()-start
        with torch.inference_mode(): video=vae.decode([result[0][0].float()])[0]
        torch.cuda.synchronize();total=time.perf_counter()-start
        if not torch.isfinite(video).all() or not torch.isfinite(result[0]).all(): raise AssertionError('Nonfinite')
        audit=control.summary()
        if audit['cross_stage_block_reuse_hits']!=sum(len(m) for m in masks.values()): raise AssertionError('Skipped-block count')
        if export:
            if not getattr(a, 'no_export_media', False):
                save_video(video,str(out/f'{variant}.mp4'),fps=16)
                from PIL import Image
                frame=video[:,video.shape[1]//2].float().cpu().clamp(-1,1)
                Image.fromarray(((frame.permute(1,2,0)+1)*127.5).round().byte().numpy()).save(out/f'{variant}.png')
            latents[variant]=result[0].float().cpu()
        return dict(variant=variant,denoise_seconds=denoise,total_seconds=total,
            vae_seconds=total-denoise,stage_seconds=result[2],audit=audit,
            peak_gib=torch.cuda.max_memory_allocated()/2**30)
    for v in variants:
        for j in range(a.warmups):
            warm=run(v,export=(j==a.warmups-1))
            if v=='baseline' and j==a.warmups-1:
                protocol['warm_baseline']=warm
                protocol['zero_denoiser_total_reduction_ceiling_percent']=100*warm['denoise_seconds']/warm['total_seconds']
                print(json.dumps(dict(warm_baseline=warm,
                    ceiling_percent=protocol['zero_denoiser_total_reduction_ceiling_percent'])),flush=True)
            (out/'progress.json').write_text(json.dumps(protocol,indent=2))
    for v in variants[1:]:
        groups=[];br=[];cr=[];reductions={'denoise':[],'total':[]}
        for g in range(a.groups):
            order=['baseline',v,v,'baseline'] if g%2==0 else [v,'baseline','baseline',v]
            rows=[run(n) for n in order];b=[r for r in rows if r['variant']=='baseline'];c=[r for r in rows if r['variant']==v]
            br+=b;cr+=c
            for k in reductions:
                reductions[k].append(100*(1-statistics.mean(r[k+'_seconds'] for r in c)/statistics.mean(r[k+'_seconds'] for r in b)))
            groups.append(rows)
            print(json.dumps(dict(size=f'{a.width}x{a.height}',variant=v,group=g,reductions={k:x[-1] for k,x in reductions.items()})),flush=True)
        protocol['entries'].append(dict(variant=v,groups=groups,
            baseline={k:aggregate([r[k+'_seconds'] for r in br]) for k in reductions},
            candidate={k:aggregate([r[k+'_seconds'] for r in cr]) for k in reductions},
            reductions={k:group_bootstrap_ci(x) for k,x in reductions.items()}))
        (out/'progress.json').write_text(json.dumps(protocol,indent=2))
    protocol['latent_relative_l2']={v:((z-latents['baseline']).square().sum()/latents['baseline'].square().sum()).sqrt().item() for v,z in latents.items()}
    protocol['complete']=True;(out/'metrics.json').write_text(json.dumps(protocol,indent=2))
    print('COMPLETE: direct reuse; shared GPU cost/quality sanity only',flush=True)
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True);p.add_argument('--width',type=int,default=832);p.add_argument('--height',type=int,default=448)
    p.add_argument('--frames',type=int,default=29)
    p.add_argument('--port',type=int,default=29951);p.add_argument('--groups',type=int,default=3)
    p.add_argument('--variants',default='fine18,fine24,cascade24,cascade28')
    p.add_argument('--factors',choices=['spatial421','temporal421','temporal221','st421','spatial221','st221'],default='spatial421')
    p.add_argument('--vae-precision',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--warmups',type=int,default=1);p.add_argument('--seed',type=int,default=20660913)
    p.add_argument('--prompt',default='A red fox walking through a snowy forest, cinematic natural light.')
    args=p.parse_args()
    if args.groups<2 or args.warmups<1: p.error('Require at least two paired groups and one warmup')
    main(args)
