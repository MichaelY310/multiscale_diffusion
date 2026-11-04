#!/usr/bin/env python3
"""Full inference-only author workflows on random local architectures."""
import argparse,json,os,sys,time,traceback,tempfile
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
import conformance
from umdiff import build,native
from umdiff.randomness import GlobalRandomStream
compare=conformance.compare
AUDITS=[]
def audit(name,pairs):
    from activation_trace import ActivationAudit
    class CanonicalBridgeAudit(ActivationAudit):
        def capture(self,side,module,index,direction,value):
            if module=='bridge/<root>':
                # Author Tensor(input_x,i_stage) vs component BoundaryResult
                # (state,stage) are explicit API encodings of the same bridge.
                if direction=='input':value={'state':value.get('input_x',value.get('state'))}
                elif hasattr(value,'state'):value=value.state
            super().capture(side,module,index,direction,value)
    cls=CanonicalBridgeAudit if name=='nami' else ActivationAudit
    result=cls(name,pairs,conformance.ARTIFACT_ROOT.parent/'layers');AUDITS.append(result);return result

def tiny_flux(guidance_embeds=False):
    from diffusers import FluxTransformer2DModel,AutoencoderKL,FluxPipeline,FlowMatchEulerDiscreteScheduler
    transformer=FluxTransformer2DModel(in_channels=16,num_layers=1,num_single_layers=1,num_attention_heads=2,attention_head_dim=16,joint_attention_dim=8,pooled_projection_dim=8,axes_dims_rope=(4,6,6),guidance_embeds=guidance_embeds).eval()
    vae=AutoencoderKL(block_out_channels=(32,32),down_block_types=('DownEncoderBlock2D',)*2,up_block_types=('UpDecoderBlock2D',)*2,latent_channels=4,norm_num_groups=32,layers_per_block=1,scaling_factor=.3611,shift_factor=.1159).eval()
    pipe=FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler(),vae=vae,transformer=transformer,text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None)
    pipe.set_progress_bar_config(disable=True)
    return pipe

def run_nami():
    from umdiff.boundaries import LearnedAffine
    from umdiff.samplers import NAMISampler
    model=build('model',{'name':'native_nami','args':{'in_channels':64,'num_layers':1,'num_single_layers':3,'num_attention_heads':1,'attention_head_dim':16,'joint_attention_dim':8,'pooled_projection_dim':8,'axes_dims_rope':[4,6,6],'stage_singleblock_index':[1,2,3]}}).eval()
    from diffusers import FluxPipeline,AutoencoderKL
    original_bridge=native.extract('nami/src/infer.py',['BridgeFlow'],dict(nn=torch.nn,torch=torch,F=torch.nn.functional))['BridgeFlow']().eval()
    bridge=LearnedAffine().eval();bridge.load_state_dict(original_bridge.state_dict())
    scheduler=native.load('nami/src/scheduling_flow_matching.py').PyramidFlowMatchEulerDiscreteScheduler()
    vae=AutoencoderKL(block_out_channels=(32,)*4,down_block_types=('DownEncoderBlock2D',)*4,up_block_types=('UpDecoderBlock2D',)*4,latent_channels=16,norm_num_groups=32,layers_per_block=1,scaling_factor=.3611,shift_factor=.1159).eval()
    pipe=FluxPipeline(scheduler,vae,model,None,None,None,None,bridge_flow=original_bridge);pipe.set_progress_bar_config(disable=True)
    prompt=torch.randn(1,3,8);pooled=torch.randn(1,8);negative=torch.randn_like(prompt);negative_pooled=torch.randn_like(pooled)
    # Diffusers activation objects are shared singletons across model and VAE.
    # Clone the graph jointly so those aliases have the same hook invocation
    # identity on BOTH sides, rather than splitting only the unified graph.
    port_model,port_vae=deepcopy((model,vae))
    trace=audit('nami',[('network',model,port_model),('bridge',original_bridge,bridge),('vae',vae,port_vae)])
    for index,scale in enumerate((1.,3.)):
        trace.begin(index);trace.side('native')
        kwargs=dict(prompt_embeds=prompt,pooled_prompt_embeds=pooled,negative_prompt_embeds=negative,negative_pooled_prompt_embeds=negative_pooled)
        expected=pipe(height=1024,width=1024,num_inference_steps=[3,3,3],guidance_scale=scale,generator=torch.Generator().manual_seed(8492),output_type='latent',**kwargs).images
        initial=torch.randn(1,16,128,128,generator=torch.Generator().manual_seed(8492))
        trace.side('unified');actual=NAMISampler(steps=[3,3,3],guidance_scale=scale)(port_model,initial,model_kwargs={**kwargs,'bridge_flow':bridge})
        compare(f'nami/full_native_pipeline/cfg{scale}',actual,expected,atol=4e-5,rtol=3e-4,scope='native_inference_latents')
        # The native decode path is verified too, with the same random local VAE.
        trace.side('native');ea=vae.decode(expected/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]
        trace.side('unified');aa=port_vae.decode(actual/port_vae.config.scaling_factor+port_vae.config.shift_factor,return_dict=False)[0]
        compare(f'nami/decoded_pixels/cfg{scale}',aa,ea,atol=2e-5,rtol=2e-4,scope='native_inference_pixels')
        trace.finish_segment()
    return {'native_compatibility':'Uses author-vendored Diffusers, whose pipeline/transformer hashes equal src copies. Missing Transformers Flax filename constant restored for import only. No model downloads. Synthetic text embeddings replace text encoders. Three stages retain fixed 16x32/64/128 latent bridge geometry and progressive single-block depth; small feature width and depth.', 'layer_api_adapter':'Bridge input_x/i_stage mapped to state/stage; BoundaryResult unwrapped to Tensor. No values modified. Model and VAE cloned jointly to preserve shared stateless SiLU module aliases.'}

def run_mrflow():
    from umdiff.superresolution import RealESRGANPixels,MrFlow
    model=build('model',{'name':'native_rrdb','args':{'num_in_ch':3,'num_out_ch':3,'num_feat':8,'num_block':2,'num_grow_ch':4,'scale':2}}).eval()
    from RealESRGAN import RealESRGAN
    from diffusers import FluxImg2ImgPipeline
    pipe=tiny_flux();refiner=FluxImg2ImgPipeline(**pipe.components);refiner.set_progress_bar_config(disable=True)
    prompt=torch.randn(1,3,8);pooled=torch.randn(1,8)
    # Execute author's main verbatim; replace only asset loaders with already
    # initialized local modules and synthetic prompt embeddings.
    pipe.encode_prompt=lambda *args,**kwargs:(prompt,pooled,torch.zeros(3,3))
    port_sr=deepcopy(model)
    trace=audit('mrflow',[('transformer',pipe.transformer,pipe.transformer),('vae',pipe.vae,pipe.vae),('sr',model,port_sr)])
    trace.begin(0);trace.side('native')
    class LocalSR(RealESRGAN):
        def __init__(self,device,scale):self.device=device;self.scale=scale;self.model=model
        def load_weights(self,path,download=False):
            if download:raise AssertionError('Downloads forbidden')
    with tempfile.TemporaryDirectory(prefix='umdiff_mrflow_') as directory:
        env=native.extract('mrflow/flux1_mrflow.py',['main'],dict(torch=torch,OUT_DIR=Path(directory),DEVICE='cpu',MODEL='synthetic-local-model',REALESRGAN_X2='synthetic-local-sr',PROMPT='synthetic',SEED=2026,LOW_SIZE=32,HIGH_SIZE=64,LOW_STEPS=12,REFINE_STEPS=1,REFINE_SIGMA=.12,GUIDANCE_SCALE=3.5,FluxPipeline=SimpleNamespace(from_pretrained=lambda *a,**k:pipe),FluxImg2ImgPipeline=FluxImg2ImgPipeline,RealESRGAN=LocalSR,direct_sigma_schedule=native.load('mrflow/mrflow_utils.py').direct_sigma_schedule))
        env['main']()
        from PIL import Image
        expected={k:np.asarray(Image.open(Path(directory)/filename)).copy() for k,filename in [('low','stage1_low.png'),('upscaled','stage2_upscaled.png'),('refined','stage3_refined.png')]}
    trace.side('unified');actual=MrFlow(low_size=32,high_size=64)(pipe,RealESRGANPixels(port_sr),refiner,prompt,pooled)
    trace.finish_segment()
    for stage in expected:compare('mrflow/full_native_workflow/'+stage,np.asarray(actual[stage]).copy(),expected[stage],atol=0,rtol=0,scope='native_inference_pixels')
    # Multi-tile SR, non-square shape, non-divisible sides, and batches >1.
    for hw in ((34,50),(64,80)):
        image=Image.fromarray(torch.randint(0,256,(*hw,3),dtype=torch.uint8).numpy())
        sr=LocalSR(torch.device('cpu'),2)
        expected=sr.predict(image,batch_size=3,patches_size=32,padding=8,pad_size=15)
        actual=RealESRGANPixels(model,patch_size=32,padding=8,batch_size=3)(image)
        compare('mrflow/sr_tiled_pixels/'+str(hw),np.asarray(actual).copy(),np.asarray(expected).copy(),atol=0,rtol=0,scope='native_inference_transition_pixels')
    return {'native_compatibility':'Actual author main function executed with local random Flux/VAE/SR and synthetic embeddings; only loaders/paths replaced. Native 12+1 steps and sigma .12, default native SR tiling, PIL quantization, VAE posterior sampling preserved. RRDB feature width8/blocks2 is a small architecture fixture, not pretrained quality. New SR source pin exactly matches MrFlow gitlink. Deprecated cached_download import replaced by a blocker.'}

def run_speed():
    from umdiff.native_models import activate
    from umdiff.samplers import SPEEDFluxSampler
    activate('speed',['utils']);source=native.load('speed/latent_image_gen.py')
    pipe=tiny_flux(guidance_embeds=True);pipe.transformer.to(dtype=torch.bfloat16);pipe.vae.to(dtype=torch.bfloat16)
    encoded=(torch.randn(1,3,8,dtype=torch.bfloat16),torch.randn(1,8,dtype=torch.bfloat16),torch.zeros(3,3,dtype=torch.bfloat16))
    trace=audit('speed',[('transformer',pipe.transformer,pipe.transformer),('vae',pipe.vae,pipe.vae)])
    for index,transform in enumerate(('dct','dwt','fft')):
        trace.begin(index);trace.side('native')
        scheduler=source.setup_scheduler(pipe,12,16,16,'cpu')
        expected=source.generate(pipe,scheduler,encoded,scales=[.5,1.],transition_times=[.5],transform=transform,H_lat=16,W_lat=16,C=4,n_steps=12,guidance=3.5,seed=8201,device='cpu')
        initial=torch.randn(1,4,8,8,generator=torch.Generator().manual_seed(8201),dtype=torch.bfloat16)
        trace.side('unified');actual=SPEEDFluxSampler(transform=transform,seed=8201)(pipe.transformer,initial,model_kwargs=dict(zip(['prompt_embeds','pooled_prompt_embeds','text_ids'],encoded)))
        compare('speed/full_native_flux/'+transform,actual,expected,atol=0,rtol=0,scope='native_inference_latents')
        with torch.no_grad():
            trace.side('unified')
            a=pipe.vae.decode(actual/.3611+.1159,return_dict=False)[0].float().clamp(-1,1)
            trace.side('native')
            b=pipe.vae.decode(expected/.3611+.1159,return_dict=False)[0].float().clamp(-1,1)
        compare('speed/decoded_float_pixels/'+transform,a,b,atol=0,rtol=0,scope='native_inference_pixels')
        compare('speed/decoded_pixels/'+transform,((a+1)*127.5).to(torch.uint8),((b+1)*127.5).to(torch.uint8),atol=0,rtol=0,scope='native_inference_pixels')
        trace.finish_segment()
    return {'native_compatibility':'Native generate loop executed unchanged with random tiny Flux and VAE; bf16 initial/model states preserved on CPU, author scheduler backend reused explicitly. All DCT/Haar/FFT transitions and final decoded pixels compared. No author training objective exists for this inference-only method.'}

RUNNERS={'nami':run_nami,'mrflow':run_mrflow,'speed':run_speed}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=RUNNERS);parser.add_argument('--output-dir',type=Path);args=parser.parse_args();torch.set_num_threads(2);torch.manual_seed(20261008);started=time.time();failure=None;details={};status='passed'
    if args.output_dir:conformance.ARTIFACT_ROOT=args.output_dir.resolve()/'artifacts'
    try:
        with torch.no_grad():details=RUNNERS[args.method]()
    except Exception:status='failed';failure=traceback.format_exc();print(failure,file=sys.stderr)
    activation=[a.close() for a in AUDITS]
    if any(a['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for k,v in a['totals'].items() if k.startswith('forward')) for a in activation):status='failed';failure=failure or 'Per-layer inference input/output bytes differ'
    import hashlib
    report=dict(method=args.method,status=status,device='cpu',gpu_used=False,seconds=time.time()-started,details=details,checks=conformance.CHECKS,source_executions=native.AUDIT,activation=activation,backward_status='not_applicable: released inference is no_grad; no released author training objective',harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),failure=failure,quality_claim='Random local model and synthetic input conformance; no paper FID or perceptual-quality guarantee.')
    directory=args.output_dir or ROOT/'evidence';directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{args.method}_inference.json';path.write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ['method','status','seconds']}));return status!='passed'
if __name__=='__main__':raise SystemExit(main())
