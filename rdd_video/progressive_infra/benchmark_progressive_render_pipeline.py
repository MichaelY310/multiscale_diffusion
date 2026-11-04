"""Actual 3-NFE execution + early parent rendering, TRAINING-REQUIRED pilot.

The generator uses an untrained lossless temporal-patch2 embedding/head.
The final low band is explicitly committed to the middle prediction. This
changes the generative process; neither renderer nor generator is trained.
All controls get identical two GPUs, fresh30blocks, VSA20 and the same codec.
Ordinary postfinal starts h-only work while the parent decoder runs.
"""
import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import sys
import time
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from sample_fastwan_native_vsa import add_fastvideo_paths, sample_three_call_dmd
from benchmark_temporal_band_memory import TemporalPatch2
from benchmark_csra_prefix_feasibility import aggregate, group_bootstrap_ci
from progressive_renderer import TemporalInnovationRenderer


def main(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '6,7':
        raise ValueError('Only currently reserved physical6/7 for this pilot')
    out=Path(a.output)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():
        raise FileExistsError(out)
    for i in (0,1):
        if torch.cuda.mem_get_info(i)[0] < 60*2**30:
            raise RuntimeError('Need60GiB free on each GPU; never evict jobs')
        torch.cuda.set_per_process_memory_fraction(.65,i)
    torch.set_num_threads(2)
    torch.manual_seed(2197)
    root=Path(a.root)
    dev=torch.device('cuda:0')
    decoder_dev=torch.device('cuda:1')
    torch.cuda.set_device(dev)
    fv=root/'third_party/FastVideo-ref'
    add_fastvideo_paths(fv,fv/'fastvideo-kernel')
    from fastvideo.configs.models.dits import WanVideoConfig
    from fastvideo.configs.pipelines.wan import FastWan2_1_T2V_480P_Config
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.fastvideo_args import FastVideoArgs
    from fastvideo.models.loader.component_loader import TransformerLoader
    from wan.modules.t5 import T5EncoderModel
    from sample_rdd_wan import save_video
    wan=root/'pretrained_models/Wan2.1-T2V-1.3B'
    text=T5EncoderModel(text_len=512,dtype=torch.float32,device=torch.device('cpu'),
        checkpoint_path=str(wan/'models_t5_umt5-xxl-enc-bf16.pth'),tokenizer_path=str(wan/'google/umt5-xxl'))
    text.model.to(dev)
    with torch.inference_mode():
        tokens=text([a.prompt],dev)[0]
        context=torch.zeros((1,512,tokens.shape[-1]),device=dev,dtype=torch.bfloat16)
        context[0,:len(tokens)]=tokens.to(torch.bfloat16)
    text.model.cpu()
    del text,tokens
    torch.cuda.empty_cache()
    os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='29947',RANK='0',WORLD_SIZE='1',LOCAL_RANK='0')
    maybe_init_distributed_environment_and_model_parallel(1,1)
    weights=root/'pretrained_models/FastWan2.1-T2V-1.3B-Diffusers/transformer'
    fa=FastVideoArgs(model_path=str(weights),num_gpus=1,dit_cpu_offload=False,
        dit_layerwise_offload=False,pipeline_config=FastWan2_1_T2V_480P_Config(dit_config=WanVideoConfig(),dit_precision='bf16'))
    fa.device=dev
    model=TransformerLoader().load(str(weights),fa).to(dtype=torch.bfloat16).eval().requires_grad_(False)
    packing=TemporalPatch2(model)
    packing.enable(True)
    code=root/'third_party/Flash-VAED-ref/models/wan/model_hybrid_aggressive.py'
    spec=importlib.util.spec_from_file_location('flash_ppr_budget',code)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with torch.cuda.device(decoder_dev):
        codec=mod.WanVAE(vae_pth=str(code.parent/'Flash_VAED_Wan.pth'),device=decoder_dev,dtype=torch.bfloat16)
    renderer=TemporalInnovationRenderer(width=a.width,depth=4).to(dev,dtype=torch.bfloat16).eval().requires_grad_(False)
    shape=(1,16,16,96,160)
    exact_reference=None
    checks={}

    @torch.inference_mode()
    def run(variant,export=False):
        nonlocal exact_reference
        torch.cuda.synchronize(dev)
        torch.cuda.synchronize(decoder_dev)
        parent=None
        parent_rgb=None
        decode_events=None
        begin=time.perf_counter()
        def decode(c):
            nonlocal decode_events
            # The generator forces FLASH for its BF16 transformer. Flash-VAED
            # also has FP32 attention, so do not leak that restriction here.
            with torch.cuda.device(decoder_dev), sdpa_kernel([
                    SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH]):
                s,e=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                s.record()
                copied=c.to(decoder_dev,non_blocking=True)
                rgb=codec.decode(copied)[0].unsqueeze(0)
                e.record()
                decode_events=(s,e)
            return rgb
        def consume(stage,c):
            nonlocal parent,parent_rgb
            if stage==1:
                parent=c
                if variant=='early':
                    parent_rgb=decode(parent)
        with contextlib.redirect_stdout(io.StringIO()),sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            final,*_=sample_three_call_dmd(model,context,shape,[(1,4),(1,2),(1,1)],
                [1000,757,522],a.seed,.8,'noise_filling',dev,stage_output_callback=consume)
        torch.cuda.synchronize(dev)
        denoise=time.perf_counter()-begin
        if variant=='flash_full':
            with torch.cuda.device(decoder_dev), sdpa_kernel([
                    SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION,
                    SDPBackend.MATH]):
                rgb=codec.decode(final.to(decoder_dev,non_blocking=True))[0].unsqueeze(0)
        else:
            h=(final[:,:,0::2]-final[:,:,1::2])/2
            if variant=='ordinary_pooled':
                parent=(final[:,:,0::2]+final[:,:,1::2])/2
            if variant!='early':
                # Decoder asynchronously produces clean RGB while current-h
                # features are prepared on the generator GPU. Strong baseline.
                parent_rgb=decode(parent)
            prepared=renderer.prepare_detail(parent,h)
            rgb=renderer.finish(parent_rgb.to(dev,non_blocking=True),prepared)
        torch.cuda.synchronize(dev)
        torch.cuda.synchronize(decoder_dev)
        total=time.perf_counter()-begin
        if not torch.isfinite(rgb).all():
            raise AssertionError('Nonfinite rendered RGB')
        if export:
            if variant=='late':
                exact_reference=rgb.cpu()
            elif variant=='early' and exact_reference is not None:
                error=float((rgb.cpu()-exact_reference).abs().max())
                checks['early_late_same_committed_codec_max_abs']=error
                if error>1e-5:
                    raise AssertionError('Pipeline changed identical codec output')
            save_video(rgb[0],str(out/(variant+'.mp4')),fps=16)
            from PIL import Image
            frame=rgb[0,:,rgb.shape[2]//2].float().cpu().clamp(-1,1)
            Image.fromarray(((frame.permute(1,2,0)+1)*127.5).round().byte().numpy()).save(out/(variant+'.png'))
        return dict(variant=variant,denoise_seconds=denoise,total_seconds=total,
            parent_decode_and_copy_gpu_seconds=decode_events[0].elapsed_time(decode_events[1])/1000 if decode_events else None)

    variants=['flash_full','late','early','ordinary_pooled']
    for v in variants:
        for j in range(2):
            row=run(v,export=j==1)
        print(json.dumps(dict(warmup=v,row=row)),flush=True)
    report=dict(complete=False,scope='Two-GPU actual3NFE timing pilot, NOT trained quality or final method',
        physical_gpus=[6,7],generator_gpu=6,decoder_gpu=7,shape=list(shape),NFE=3,VSA_keep=.2,
        all30_blocks_fresh=True,packing=packing.audit,renderer_trained=False,
        generative_contract='Middle low-band committed; fine detail computed freshly; requires conditional distillation',
        controls='Same temporal-patch2 generator and same codec; late/ordinary_pooled overlap h preparation with parent decode; no artificial global dependency',
        checks=checks,groups=[])
    for g in range(4):
        order=variants+variants[::-1] if g%2==0 else variants[::-1]+variants
        rows=[run(v) for v in order]
        report['groups'].append(rows)
        print(json.dumps(dict(group=g,seconds={v:statistics.mean(r['total_seconds'] for r in rows if r['variant']==v) for v in variants})),flush=True)
        (out/'progress.json').write_text(json.dumps(report,indent=2))
    report['times']={v:{k:aggregate([r[k+'_seconds'] for rows in report['groups'] for r in rows if r['variant']==v]) for k in ('denoise','total')} for v in variants}
    report['early_vs']={v:group_bootstrap_ci([100*(1-statistics.mean(r['total_seconds'] for r in rows if r['variant']=='early')/statistics.mean(r['total_seconds'] for r in rows if r['variant']==v)) for rows in report['groups']]) for v in ('flash_full','late','ordinary_pooled')}
    report['complete']=True
    (out/'metrics.json').write_text(json.dumps(report,indent=2))
    torch.distributed.destroy_process_group()
    print('COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=64)
    p.add_argument('--seed',type=int,default=20261003)
    p.add_argument('--prompt',default='A red fox walking through a snowy forest, cinematic natural light.')
    main(p.parse_args())
