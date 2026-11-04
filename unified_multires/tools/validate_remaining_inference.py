#!/usr/bin/env python3
"""Previously missing author inference routes, random local CPU models only."""
import argparse, copy, hashlib, json, os, sys, tempfile, time, traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from umdiff import build,native
from activation_trace import ActivationAudit,byte_view
import conformance


def same(name,a,b):
    if isinstance(a,np.ndarray):a=torch.from_numpy(a.copy())
    if isinstance(b,np.ndarray):b=torch.from_numpy(b.copy())
    exact=a.shape==b.shape and a.dtype==b.dtype and torch.equal(byte_view(a),byte_view(b))
    if not exact:raise AssertionError(f'{name}: shape/dtype/bytes differ; max_abs={float((a.float()-b.float()).abs().max())}')
    return dict(name=name,shape=list(a.shape),dtype=str(a.dtype),byte_exact=True)


def qwen(out):
    from diffusers import QwenImagePipeline,QwenImageImg2ImgPipeline,QwenImageTransformer2DModel,AutoencoderKLQwenImage,FlowMatchEulerDiscreteScheduler
    from umdiff.superresolution import RealESRGANPixels,MrFlow
    model=QwenImageTransformer2DModel(in_channels=16,out_channels=4,num_layers=1,
        attention_head_dim=16,num_attention_heads=2,joint_attention_dim=8,axes_dims_rope=(4,6,6)).eval()
    vae=AutoencoderKLQwenImage(base_dim=16,z_dim=4,dim_mult=[1,2,2,2],num_res_blocks=1,
        latents_mean=[0.]*4,latents_std=[1.]*4).eval()
    pipe=QwenImagePipeline(FlowMatchEulerDiscreteScheduler(),vae,None,None,model)
    pipe.set_progress_bar_config(disable=True)
    pos=torch.randn(1,3,8);neg=torch.randn(1,3,8);mask=torch.ones(1,3,dtype=torch.long)
    pipe.encode_prompt=lambda prompt=None,*args,**kwargs: (neg,mask) if prompt=='negative' else (pos,mask)
    sr=build('model',{'name':'native_rrdb','args':dict(num_in_ch=3,num_out_ch=3,num_feat=8,num_block=2,num_grow_ch=4,scale=2)}).eval()
    from RealESRGAN import RealESRGAN
    class LocalSR(RealESRGAN):
        def __init__(self,device,scale):self.device=device;self.scale=scale;self.model=sr
        def load_weights(self,path,download=False):
            if download:raise AssertionError('Downloads forbidden')
    audit=ActivationAudit('mrflow_qwen',[('transformer',model,model),('vae',vae,vae),('sr',sr,sr)],out/'layers')
    rows=[]
    for index,cfg in enumerate((1.,4.)):
        audit.begin(index);audit.side('native')
        with tempfile.TemporaryDirectory(prefix='umdiff_qwen_') as directory:
            env=native.extract('mrflow/qwen_image_mrflow.py',['main'],dict(
                torch=torch,OUT_DIR=Path(directory),DEVICE='cpu',MODEL='local-random',REALESRGAN_X2='local-random',
                PROMPT='positive',NEGATIVE_PROMPT='negative',SEED=2026,LOW_SIZE=32,HIGH_SIZE=64,LOW_STEPS=12,
                REFINE_STEPS=1,REFINE_SIGMA=.12,TRUE_CFG_SCALE=cfg,
                QwenImagePipeline=SimpleNamespace(from_pretrained=lambda *a,**k:pipe),QwenImageImg2ImgPipeline=QwenImageImg2ImgPipeline,
                RealESRGAN=LocalSR,direct_sigma_schedule=native.load('mrflow/mrflow_utils.py').direct_sigma_schedule))
            env['main']()
            expected={k:np.asarray(Image.open(Path(directory)/f'stage{i}_{filename}.png')).copy()
                for k,i,filename in [('low',1,'low'),('upscaled',2,'upscaled'),('refined',3,'refined')]}
        audit.side('unified')
        refiner=QwenImageImg2ImgPipeline(**pipe.components);refiner.set_progress_bar_config(disable=True)
        actual=MrFlow(low_size=32,high_size=64)(pipe,RealESRGANPixels(sr),refiner,conditioning=dict(
            prompt=None,negative_prompt=None,prompt_embeds=pos,prompt_embeds_mask=mask,
            negative_prompt_embeds=neg,negative_prompt_embeds_mask=mask,true_cfg_scale=cfg))
        for k in expected:rows.append(same(f'qwen/cfg{cfg}/{k}',np.asarray(actual[k]),expected[k]))
        audit.finish_segment()
    return dict(checks=rows,activation=audit.close(),scope='Author Qwen main; local random Qwen transformer/VAE/RRDB, synthetic prompt embeddings, 32→64 pixels,12+1 steps. No pretrained-quality certification.')


def pixelgen(out):
    from umdiff.native_models import activate
    from umdiff.compat.speed_routes import SPEEDPixelGenSampler,attach_pixelgen_rope
    activate('pixelgen',['src']);activate('speed',['utils'])
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.models.layers.rope import precompute_freqs_cis_ex2d
    from src.models.autoencoder.pixel import PixelAE
    from src.models.autoencoder.base import fp2uint8
    source=native.load('speed/pixel_image_gen.py')
    from validate_training import nonzero_output
    original=nonzero_output(JiT_T2I(input_size=32,patch_size=4,in_channels=3,num_groups=2,
        hidden_size=32,num_blocks=2,num_text_blocks=1,txt_embed_dim=8,txt_max_length=3,bottleneck_dim=16)).eval()
    port=copy.deepcopy(original)
    original.fetch_pos=__import__('types').MethodType(source._pi_fetch_pos_factory(8),original);original.precompute_pos={}
    attach_pixelgen_rope(port,8,precompute_freqs_cis_ex2d)
    ae=PixelAE(scale=.83,shift=.13);pae=copy.deepcopy(ae)
    audit=ActivationAudit('speed_pixelgen',[('network',original,port),('vae',ae,pae)],out/'layers')
    cond=torch.randn(1,3,8);uncond=torch.randn_like(cond);rows=[]
    cases=[(tr,scales,thr,shift) for tr in ('dct','dwt','fft')
        for scales,thr,shift in [([.5,1.],[.5],1.),([.25,.5,1.],[.7,.3],3.)]]
    for index,(tr,scales,thr,shift) in enumerate(cases):
        audit.begin(index);audit.side('native')
        expected=source.generate(original,cond,uncond,scales=scales,transition_times=thr,transform=tr,
            height=32,width=32,n_steps=12,guidance=4.,timeshift=shift,seed=9231,device='cpu',trained_grid=8)
        decoded=ae.decode(expected);pixels=fp2uint8(decoded)
        audit.side('unified');initial=torch.randn(1,3,round(32*scales[0]),round(32*scales[0]),generator=torch.Generator().manual_seed(9231))
        actual=SPEEDPixelGenSampler(full_hw=(32,32),scales=scales,transitions=thr,steps=12,
            transform=tr,seed=9231,guidance=4.,timeshift=shift,trained_grid=8)(port,initial,model_kwargs={'cond':cond,'uncond':uncond})
        pdecoded=pae.decode(actual);ppixels=torch.clip_((pdecoded+1)*127.5+.5,0,255).to(torch.uint8)
        rows.extend([same(f'{tr}/{scales}/{name}',a,b) for name,a,b in
            [('state',actual,expected),('float_pixels',pdecoded,decoded),('uint8',ppixels,pixels),('position',port.pos_embed,original.pos_embed)]])
        audit.finish_segment()
    return dict(checks=rows,activation=audit.close(),scope='Actual pinned PixelGen JiT_T2I, PixelAE and SPEED generate; random nonzero reduced model, synthetic embeddings, two/three scales,DCT/DWT/FFT. Native CUDA autocast disables itself on CPU; FP32 here.')


def wan(out):
    import ast,math
    from diffusers.configuration_utils import ConfigMixin,register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    from umdiff.native_models import activate
    from umdiff.compat.speed_routes import SPEEDWanSampler
    activate('speed',['utils']);source=native.load('speed/latent_video_gen.py')
    class CPUAMP:
        @staticmethod
        def autocast(dtype=None,enabled=True):
            return torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled and dtype!=torch.float32)
    def attention(q,k,v,k_lens=None,**kwargs):
        q=q.to(v.dtype);k=k.to(v.dtype)
        mask=None
        if k_lens is not None:mask=torch.arange(k.shape[1])[None,None,None,:]<k_lens[:,None,None,None]
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
    path=ROOT/'sources/wan21/wan/modules/model.py'
    names=[n.name for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    env=native.extract('wan21/wan/modules/model.py',names,dict(torch=torch,nn=torch.nn,math=math,amp=CPUAMP,
        flash_attention=attention,ConfigMixin=ConfigMixin,register_to_config=register_to_config,ModelMixin=ModelMixin,
        T5_CONTEXT_TOKEN_NUMBER=512,FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER=514))
    native.statements('wan21/wan/modules/model.py',232,235,env)
    env['WAN_CROSSATTENTION_CLASSES']={'t2v_cross_attn':env['WanT2VCrossAttention'],'i2v_cross_attn':env['WanI2VCrossAttention']}
    class CPUWan(torch.nn.Module):
        def __init__(self):
            super().__init__();self.network=env['WanModel'](text_len=8,in_dim=4,out_dim=4,dim=32,ffn_dim=64,
                freq_dim=16,text_dim=8,num_heads=2,num_layers=1)
        def forward(self,*args,**kwargs):
            with torch.autocast('cpu',dtype=torch.bfloat16):return self.network(*args,**kwargs)
    original=CPUWan().eval().requires_grad_(False)
    port=build('model',{'name':'native_wan21_cpu','args':dict(text_len=8,in_dim=4,out_dim=4,dim=32,ffn_dim=64,freq_dim=16,text_dim=8,num_heads=2,num_layers=1)})
    port.load_state_dict(original.network.state_dict())
    # A real tiny Wan VAE is independently decoded on both sides. This is not
    # the unpublished author's pretrained asset, and the source PNG/MP4 writer
    # is deliberately kept outside float-pixel parity.
    from diffusers import AutoencoderKLWan
    ae=AutoencoderKLWan(base_dim=32,decoder_base_dim=32,z_dim=4,dim_mult=[1,1,1,1],
        num_res_blocks=1,latents_mean=[0.]*4,latents_std=[1.]*4).eval();pae=copy.deepcopy(ae)
    audit=ActivationAudit('speed_wan',[('network',original.network,port),('vae',ae,pae)],out/'layers')
    cond=[torch.randn(3,8)];uncond=[torch.randn(3,8)];rows=[]
    cases=[(tr,hw) for tr in ('dct','dwt','fft') for hw in ((8,12),(10,14))]
    for index,(tr,hw) in enumerate(cases):
        audit.begin(index);audit.side('native');scheduler=source.setup_scheduler(12,3.,'cpu')
        expected=source.generate(original,scheduler,cond,uncond,scales=[.5,1.],transition_times=[.5],transform=tr,
            H_lat=hw[0],W_lat=hw[1],T_lat=2,C=4,n_steps=12,guidance=5.,seed=8201,device='cpu')
        decoded=ae.decode(expected[None],return_dict=False)[0]
        audit.side('unified');initial=torch.randn(4,2,round(hw[0]*.5),round(hw[1]*.5),
            generator=torch.Generator().manual_seed(8201),dtype=torch.bfloat16)
        actual=SPEEDWanSampler(full_hw=hw,steps=12,transform=tr,seed=8201)(port,initial,
            model_kwargs={'ctx_cond':cond,'ctx_uncond':uncond})
        pdecoded=pae.decode(actual[None],return_dict=False)[0]
        rows.extend([same(f'{tr}/{hw}/state',actual,expected),same(f'{tr}/{hw}/float_pixels',pdecoded,decoded)])
        audit.finish_segment()
    return dict(checks=rows,activation=audit.close(),scope='Actual native Alibaba WanModel plus SPEED author video generate, reduced random model, CPU BF16 attention/device compatibility explicitly substituted; even/odd coarse grids,rectangular video,all transforms. Float decode uses random Diffusers Wan VAE; native pretrained Wan VAE/MP4 not certified.')


RUNNERS={'qwen':qwen,'pixelgen':pixelgen,'wan':wan}
def main():
    p=argparse.ArgumentParser();p.add_argument('route',choices=RUNNERS);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(20261012);np.random.seed(20261012)
    started=time.time();report=dict(route=a.route,gpu_used=False,formal_training=False,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),backward='N/A: no released author training objective')
    try:
        with torch.no_grad():report.update(RUNNERS[a.route](a.output))
        act=report['activation'];assert not act['structural_mismatches']
        assert all(v['pairs']==v['byte_exact'] for k,v in act['totals'].items() if k.startswith('forward'))
        report['status']='passed'
    except Exception:report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
    report.update(seconds=time.time()-started,source_executions=native.AUDIT,cuda_initialized=torch.cuda.is_initialized())
    (a.output/(a.route+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('route','status','seconds')}),flush=True)
    return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
