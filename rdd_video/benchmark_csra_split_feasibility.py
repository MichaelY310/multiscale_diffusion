"""Complete two-reader FastWan cost screen, including actual resident VAE decode.

UNTRAINED video readers and non-RDD-trained FastWan. NOT a video-quality result.
All weights, output shapes, VSA gates/coarse outputs and offload policies match.
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

CASES={
    '832_spatial':((1,16,8,56,104),[(4,1),(2,1),(1,1)]),
    '832_st':((1,16,8,56,104),[(4,4),(2,2),(1,1)]),
    '1280_spatial':((1,16,8,96,160),[(4,1),(2,1),(1,1)]),
    '1280_st':((1,16,8,96,160),[(4,4),(2,2),(1,1)]),
}


def main(a):
    root,out=Path(a.root).resolve(),Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Preserve completed result')
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if visible not in ('0','1','2','3'):raise ValueError('Exactly one physical GPU0-3 required')
    device=torch.device('cuda:0');torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if free<70*2**30:raise RuntimeError('Need70GiB free; do not displace other processes')
    torch.cuda.set_per_process_memory_fraction(.70,device)
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(61560916);torch.cuda.manual_seed_all(61560916)
    fv=root/'third_party/FastVideo-ref';add_fastvideo_paths(fv,fv/'fastvideo-kernel')
    from attention_analysis.fastwan_native_attention import install_projected_cache_class_patch
    from attention_analysis.csra_prefix_video_bridge import VideoConditionalPrefixBridge
    from attention_analysis.csra_split_video_bridge import VideoSpanSamplerProxy
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
    text.model.to(device)
    with torch.inference_mode():
        tokens=text([a.prompt],device)[0]
        context=torch.zeros((1,512,tokens.shape[-1]),device=device,dtype=torch.bfloat16)
        if len(tokens)>512:raise ValueError('Prompt length')
        context[0,:len(tokens)]=tokens.to(torch.bfloat16)
    text.model.cpu();del text,tokens;torch.cuda.empty_cache()
    install_projected_cache_class_patch(WanTransformerBlock_VSA)
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT=str(a.port),RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    maybe_init_distributed_environment_and_model_parallel(1,1)
    weights=root/'pretrained_models/FastWan2.1-T2V-1.3B-Diffusers/transformer'
    fa=FastVideoArgs(model_path=str(weights),num_gpus=1,dit_cpu_offload=False,dit_layerwise_offload=False,
        pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(),dit_precision='bf16'))
    fa.device=device
    original=TransformerLoader().load(str(weights),fa).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    fine=TransformerLoader().load(str(weights),fa).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    assert all(type(b).__name__=='WanTransformerBlock_VSA' and hasattr(b,'to_gate_compress')
        for model in (original,fine) for b in model.blocks)
    h,heads=original.blocks[0].hidden_dim,original.blocks[0].num_attention_heads
    readers=[VideoConditionalPrefixBridge(h,heads).to(device).eval().requires_grad_(False) for _ in range(2)]
    vae=WanVAE(z_dim=16,vae_pth=str(wan/'Wan2.1_VAE.pth'),dtype=torch.float32,device=device)
    shape,factors=CASES[a.case]
    protocol=dict(version='csra_split_complete_video_cost_v1',complete=False,case=a.case,gpu=visible,
        warning='Random UNTRAINED video readers. FastWan is NOT RDD-trained. Cost feasibility, not trained-video quality.',
        model=str(weights),hidden=h,heads=heads,layers=len(original.blocks),full_shape=shape,
        factors=factors,frames=29,fps=16,NFE=3,t=[1000,757,522],seed=20660913,prompt=a.prompt,
        transition='noise_filling',vsa_sparsity=.8,native_gates_and_coarse_output=True,
        source='middle original block13 full hidden',split_spans_zero_based=[[0,3],[6,9]],
        two_models_two_readers_and_vae_resident_in_all_arms=True,offload=False,TF32=False,
        bridge_affine='FP32',bridge_attention_mlp='BF16',bridge_SDPA='FLASH_ATTENTION only',
        vae='FP32 actual decode of each trajectory; not a separately added constant',
        excludes='T5/model load/compile warmup/MP4 encoding and disk I/O',
        warmups=a.warmups,groups=a.groups,entries=[])
    for p in (Path(__file__),root/'rdd_video/attention_analysis/csra_split_video_bridge.py',
              root/'rdd_video/attention_analysis/csra_prefix_video_bridge.py',
              root/'rdd_video/sample_fastwan_native_vsa.py',fv/'fastvideo/models/dits/wanvideo.py'):
        protocol.setdefault('source_sha256',{})[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
    def sample(model,controller=None):
        with contextlib.redirect_stdout(io.StringIO()),torch.inference_mode(),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            return sample_three_call_dmd(model,context,shape,factors,[1000,757,522],20660913,.8,
                'noise_filling',device,attention_controller=controller,boundary_executor='native',return_breakdown=True)
    ref=sample(original)[0];duplicate=sample(fine)[0]
    protocol['duplicate_max_abs']=(ref.float()-duplicate.float()).abs().max().item();del duplicate
    proxy=VideoSpanSamplerProxy(original,fine,readers)
    wrapped=sample(proxy,proxy)[0];torch.cuda.synchronize()
    protocol['baseline_wrapper_max_abs']=(ref.float()-wrapped.float()).abs().max().item();proxy.audit()
    if protocol['duplicate_max_abs']!=0 or protocol['baseline_wrapper_max_abs']!=0:raise AssertionError('Baseline parity')
    del ref,wrapped
    print('PASS native/copy/no-op trajectory exact parity',flush=True)
    def run(variant,export=False):
        proxy.reset(variant);torch.cuda.reset_peak_memory_stats(device);torch.cuda.synchronize()
        start=time.perf_counter();result=sample(proxy,proxy);torch.cuda.synchronize()
        denoise=time.perf_counter()-start
        with torch.inference_mode():video=vae.decode([result[0][0].float()])[0]
        torch.cuda.synchronize();total_elapsed=time.perf_counter()-start
        audit=proxy.audit()
        if not torch.isfinite(result[0]).all().item() or not torch.isfinite(video).all().item():raise AssertionError('Nonfinite output')
        if export:save_video(video,str(out/f'{variant}_UNTRAINED_COST_ONLY.mp4'),fps=16)
        row=dict(variant=variant,denoise_seconds=denoise,total_seconds=total_elapsed,
            vae_seconds=total_elapsed-denoise,stage_seconds=result[2],breakdown=result[4],audit=audit,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
        del result,video
        return row
    for variant in ('prefix6','split3_3'):
        for _ in range(a.warmups):run('baseline');run(variant)
        groups=[];base=[];candidate=[];reductions={'denoise':[],'total':[]}
        for g in range(a.groups):
            order=['baseline',variant,variant,'baseline'] if g%2==0 else [variant,'baseline','baseline',variant]
            rows=[run(v) for v in order]
            br=[r for r in rows if r['variant']=='baseline'];cr=[r for r in rows if r['variant']==variant]
            base+=br;candidate+=cr
            for key in reductions:
                b=statistics.mean(r[key+'_seconds'] for r in br);c=statistics.mean(r[key+'_seconds'] for r in cr)
                reductions[key].append(100*(1-c/b))
            groups.append(dict(index=g,order=order,rows=rows))
            print(f'{a.case} {variant} group{g+1}/{a.groups} denoise reduction={reductions["denoise"][-1]:.2f}% total={reductions["total"][-1]:.2f}%',flush=True)
        entry=dict(variant=variant,groups=groups,
            baseline={k:aggregate([r[k+'_seconds'] for r in base]) for k in ('denoise','vae','total')},
            candidate={k:aggregate([r[k+'_seconds'] for r in candidate]) for k in ('denoise','vae','total')},
            paired_reduction_ci95={k:group_bootstrap_ci(v) for k,v in reductions.items()},
            bridge_seconds=aggregate([r['audit']['bridge_seconds'] for r in candidate]))
        protocol['entries'].append(entry);(out/'progress.json').write_text(json.dumps(protocol,indent=2)+'\n')
    for v in ('baseline','prefix6','split3_3'):run(v,export=True)
    protocol['complete']=True;(out/'metrics.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('COMPLETE cost only; no trained-video quality claim',flush=True)
    torch.distributed.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True);p.add_argument('--case',choices=CASES,required=True)
    p.add_argument('--port',type=int,default=29760);p.add_argument('--warmups',type=int,default=2)
    p.add_argument('--groups',type=int,default=8)
    p.add_argument('--prompt',default='A red fox walking through a snowy forest, cinematic natural light.')
    main(p.parse_args())
