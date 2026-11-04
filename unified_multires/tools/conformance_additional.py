"""Additional official source conformance cases."""
import sys,math,tempfile,os
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
import torch.nn.functional as F
from umdiff import native
ROOT=native.ROOT

def install(compare,train_compare,fixture):
    globals().update(compare=compare,train_compare=train_compare,fixture=fixture)
    return {name[4:]:value for name,value in globals().items() if name.startswith('run_')}

def run_dctdiff():
    import cv2
    from PIL import Image
    from umdiff.dctcodec import DCTYCbCr
    from umdiff.paths import VP
    from torch.utils.data import Dataset
    utility=native.load('dctdiff/DCT_utils.py')
    namespace=dict(torch=torch,np=np,cv2=cv2,os=os,Image=Image,Dataset=Dataset,
                   transforms=SimpleNamespace(RandomHorizontalFlip=lambda:lambda image:image))
    namespace.update({key:getattr(utility,key) for key in ('zigzag_order','reverse_zigzag_order','split_into_blocks','dct_transform')})
    dataset_class=native.extract('dctdiff/datasets.py',['DCT_4YCbCr'],namespace)['DCT_4YCbCr']
    decoder=native.extract('dctdiff/utils.py',['DCT_to_RGB'],dict(np=np,cv2=cv2,idct_transform=utility.idct_transform,combine_blocks=utility.combine_blocks))['DCT_to_RGB']
    torch.manual_seed(43);pixels=torch.randint(0,256,(2,3,32,32),dtype=torch.uint8)
    for frequencies in (4,16):
        rep=DCTYCbCr(32,4,frequencies,256.)
        with tempfile.TemporaryDirectory(prefix='umdiff_synthetic_') as directory:
            folder=Path(directory)/'synthetic';folder.mkdir()
            for i,image in enumerate(pixels):Image.fromarray(image.permute(1,2,0).numpy()).save(folder/f'{i}.png')
            dataset=dataset_class(directory,img_sz=32,tokens=16,low_freqs=frequencies,block_sz=4,Y_bound=256.)
            dataset.img_paths=sorted(dataset.img_paths)
            expected=torch.stack([dataset[i] for i in range(2)])
        actual=rep.encode(pixels)
        compare('dctdiff/encoded_tokens',actual,expected,atol=0,rtol=0,scope='native_representation')
        expected_pixels=np.stack([decoder(value.numpy(),tokens=16,low_freqs=frequencies,block_sz=4,reverse_order=utility.reverse_zigzag_order(4),resolution=32,Y_bound=256.) for value in expected])
        compare('dctdiff/decoded_pixels',rep.decode(actual).permute(0,2,3,1),expected_pixels,atol=0,rtol=0,scope='native_pixels')
    sde=native.load('dctdiff/sde.py');_,_,times=fixture();clean=torch.randn(3,16,24);noise=torch.randn_like(clean)
    for scale in (1.,4.,16.):
        original=sde.VPSDE(SNR_scale=scale);path=VP(snr_scale=scale)
        mean,std=original.marginal_prob(clean,times);out=path.sample(clean,times,noise)
        compare('dctdiff/vp_state',out.state,mean+std[:,None,None]*noise,atol=0,rtol=0)
    sys.path.insert(0,str(ROOT/'sources/dctdiff'))
    from libs.uvit import UViT
    backbone=UViT(embed_dim=32,depth=2,num_heads=4,tokens=16,low_freqs=4,num_classes=-1)
    original=sde.VPSDE(SNR_scale=4.)
    score=sde.ScoreModel(backbone,'noise_pred',original)
    torch.manual_seed(47);loss=sde.LSimple(score,clean,reweight=torch.linspace(.5,1.5,24))
    torch.manual_seed(47);times=torch.rand(3);eps=torch.randn_like(clean)
    sample=VP(snr_scale=4.).sample(clean,times,eps)
    pred=backbone(sample.state,times*999)
    expected=((eps-pred).square()*torch.linspace(.5,1.5,24)).flatten(1).mean(1)
    compare('dctdiff/native_loss',expected,loss,atol=0,rtol=0,scope='native_training_loss')
    train_compare('dctdiff/native_uvit',backbone,sample.state,sample.state,times*999,eps,eps)

def run_dimr():
    import warnings
    warnings.filterwarnings('ignore',message='Using slower tdp_torch')
    os.environ['K_DIFFUSION_USE_COMPILE']='0';os.environ['USE_FLASH_2']='0'
    sys.path.insert(0,str(ROOT/'sources/dimr'))
    from libs.core.blocks import MRModel
    from umdiff.multiscale import dimr_targets
    author=native.load('dimr/sde.py')
    x,eps,t=fixture();x=x[:,:2];eps=eps[:,:2]
    for size in (16,17):
        value=torch.randn(3,2,size,size)
        compare('dimr/multiscale_targets',dimr_targets(value,3),author.multi_scale_targets(value,3,True),atol=0,rtol=0)
    def cfg(block,dim,size,ratio):return SimpleNamespace(block_type=block,dim=dim,hidden_dim=2*dim,num_attention_heads=4,num_blocks=1,max_height=size,max_width=size,image_input_ratio=1,input_feature_ratio=ratio,final_kernel_size=3,kernel_size=3,dropout_prob=0)
    backbone=MRModel(SimpleNamespace(channels=2,num_classes=1001,block_grad_to_lowres=False,norm_type='TDRMSN',stage_configs=[cfg('TransformerBlock',32,8,2),cfg('ConvNeXtBlock',16,16,1)]))
    # Source-native network with real TDRMSNorm, coarse/fine feature exchange.
    class Adapter(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,x,t):return self.net(x,log_snr=t,y=torch.zeros(x.shape[0],dtype=torch.long))
    model=Adapter(backbone)
    targets=author.multi_scale_targets(eps,2,True)
    target=[targets[8],targets[16]]
    train_compare('dimr/native_multires_network',model,x,x,t,target,target)

def run_mdm():
    from umdiff.multiscale import snr_shift,mdm_posterior
    source=native.load('mdm/ml-mdm-matryoshka/ml_mdm/samplers.py')
    cfg=source.SamplerConfig(num_diffusion_steps=16,prediction_type=source.PredictionType.V_PREDICTION,schedule_type=source.ScheduleType.DEEPFLOYD,schedule_shifted=True,schedule_shifted_power=1.)
    sampler=source.NestedSampler(cfg)
    x,eps,t=fixture();g=torch.tensor([.1,.4,.8]).view(-1,1,1,1);gp=(g+.1).clamp(max=.99)
    for scale in (1.,2.,4.):compare('mdm/snr_shift',snr_shift(g,scale),sampler.get_schedule_shifted(g,scale),atol=0,rtol=0)
    for kind,enum in (('v',source.PredictionType.V_PREDICTION),('epsilon',source.PredictionType.DDPM)):
        state=g.sqrt()*x+(1-g).sqrt()*eps
        target=g.sqrt()*eps-(1-g).sqrt()*x if kind=='v' else eps
        compare('mdm/target',target,source.Sampler.get_prediction_targets(sampler,x,eps,g,gp,enum),atol=0,rtol=0)
        for eta in (None,0.,.7):
            actual=mdm_posterior(state,target,g,gp,kind,eps,eta)
            expected=sampler.get_prediction_xt_last(state,target,g,gp,prediction_type=enum,need_noise=True,ddim_eta=eta,input_noise=eps)
            compare('mdm/posterior',actual,expected,atol=0,rtol=0,scope='native_inference_update')

def run_diffusability():
    import einops
    from umdiff.regularizers import ScaleEquivariance
    # Execute the exact class method with pixel reconstruction and no LPIPS.
    import ast
    path=ROOT/'sources/diffusability/src/training/loss.py';tree=ast.parse(path.read_text())
    method=next(n for c in tree.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef) and n.name=='maybe_compute_scale_equiv_reg')
    class EasyDict(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
        @staticmethod
        def init_recursively(value):return EasyDict(value)
    env=dict(torch=torch,np=np,F=F,einops=einops,EasyDict=EasyDict)
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),env)
    from umdiff.models import TinyConv
    decoder=TinyConv(3,16)
    x,eps,t=fixture();images=x[:,None];latents=eps[:,None]
    def decode(z):return decoder(z.flatten(0,1),torch.zeros(len(z))).unflatten(0,z.shape[:2])
    def native_net(a,b,c,*,latents,**kw):return decode(latents)
    for mode in ('mean','bil'):
        cfg=SimpleNamespace(scale_equiv_reg=SimpleNamespace(weight=.3,scale_factors=[.5],resample_strategy=mode,temporal_scale_factors=[],ignore_lpips=True,cut_dct2d_high_freqs=SimpleNamespace(max_cut_ratio=0.)))
        subject=SimpleNamespace(cfg=cfg,compute_rec_losses=lambda a,b,**kw:EasyDict(rec=(a-b).square().flatten(1).mean(1)))
        expected=env['maybe_compute_scale_equiv_reg'](subject,native_net,EasyDict(total=torch.zeros(3)),latents,images,None,None)
        actual=ScaleEquivariance(2,mode,.3)(decode,latents,images)
        compare('diffusability/scale_equivariance',actual,expected.total,atol=0,rtol=0,scope='native_codec_training_loss')
        ga=torch.autograd.grad(actual.mean(),tuple(decoder.parameters()),retain_graph=True)
        gb=torch.autograd.grad(expected.total.mean(),tuple(decoder.parameters()),retain_graph=True)
        compare('diffusability/codec_gradients',ga,gb,atol=0,rtol=0,scope='native_codec_training_gradients')

def run_ssd():
    sys.path.insert(0,str(ROOT/'sources/ssd'))
    from guided_diffusion import scale_space_diffusion as author
    from guided_diffusion.ssd_math_util import BilinearOp
    from umdiff.operators import BilinearOperator
    from umdiff.ssd import ScaleSpacePosterior
    from umdiff.paths import DiscreteVP
    # Singleton distributed shim: removes synchronization only. Source math
    # functions and model calls remain unchanged; no real process group needed.
    author.dist=SimpleNamespace(get_rank=lambda:0,barrier=lambda:None)
    config={'resolution_schedule':'equal','t_res_sampling_mode':'one_t_one_res','multires_training_mode':'flexi_unet','unet_res_layer_map':{8:8,16:16},'scale_min':8}
    source=author.ScaleSpaceDiffusion(betas=np.linspace(.001,.02,8),model_mean_type=author.ModelMeanType.START_X_RS,model_var_type=author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,ssd_config=config,resolutions_array=[8,16],mse_loss_weight_type='constant')
    for op in source.ops.values():op.H_out,op.W_out=op.size_out
    x,eps,t=fixture();path=DiscreteVP(betas=np.linspace(.001,.02,8))
    for index in (1,4,6):
        previous=max(index-1,0);rt=source.get_resolution(index);rs=source.get_resolution(previous)
        clean_t=source.cummulative_ops[index](x)
        clean_s=source.cummulative_ops[previous](x) if previous else x
        noise=torch.randn_like(clean_t);times=torch.full((3,),index,dtype=torch.long)
        state=source.q_sample(clean_t,times,noise=noise)
        compare('ssd/forward',path.sample(clean_t,times,noise).state,state,atol=0,rtol=0)
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.conv=torch.nn.Conv2d(3,3,3,padding=1)
            def forward(self,state,t,cond_rs=None):
                self.last=state;out=self.conv(state)
                return F.interpolate(out,(rs,rs),mode='bilinear',align_corners=False,antialias=True) if out.shape[-1]!=rs else out
        model=Model()
        loss=source.training_losses(model,x,times,None,noise=noise)['loss']
        pre,_=source.preprocess_input(state,t=times,s=(times-1).clamp(min=0),rt=rt,rs=rs)
        pred=model(pre,times,None)
        compare('ssd/native_loss',(pred-clean_s).square().flatten(1).mean(1),loss,atol=0,rtol=0,scope='native_training_loss')
        torch.manual_seed(53);innovation=torch.randn(3,3,rs,rs)
        torch.manual_seed(59)
        # Intercept raw innovations; preserve native power-iteration draw.
        from umdiff.randomness import NoiseTape
        if rs==rt:
            with NoiseTape([innovation]).patch_randn_like():expected_mean,expected_noise=source.q_posterior_mean_variance(clean_s,state,times)
        else:
            original_sampler=author.sample_from_simplified_sigma_batched
            with patch.object(author,'sample_from_simplified_sigma_batched',lambda **kw:original_sampler(**kw,xi=innovation)):
                expected_mean,expected_noise=source.q_posterior_mean_variance(clean_s,state,times)
        torch.manual_seed(59)
        mean,noise_result=ScaleSpacePosterior()(state,clean_s,source.sqrt_alphas_cumprod[previous],source.sqrt_alphas_cumprod[index],source.sqrt_one_minus_alphas_cumprod[previous],source.sqrt_one_minus_alphas_cumprod[index],BilinearOperator((rs,rs),(rt,rt)),innovation)
        compare('ssd/posterior_mean',mean,expected_mean,atol=2e-6)
        compare('ssd/posterior_innovation',noise_result,expected_noise,atol=4e-6,rtol=5e-5,scope='native_inference_update')
    from umdiff.native_models import ssd
    model=ssd(image_size=16,in_channels=3,model_channels=32,out_channels=3,num_res_blocks=1,attention_resolutions=(2,),channel_mult=(1,2),unet_res_layer_map={8:8,16:16})
    # Nonzero initialization exercises the complete source FlexiUNet.
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim>=2 and not torch.any(p):p.normal_(std=.01)
    class Wrapped(torch.nn.Module):
        def __init__(self,net):super().__init__();self.net=net
        def forward(self,state,t):return self.net(state,t,None)
    train_compare('ssd/native_flexiunet',Wrapped(model),x,x,t,x,x)
