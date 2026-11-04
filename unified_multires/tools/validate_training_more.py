"""Per-repository full CPU training fixtures; imported by validate_training."""
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from umdiff import build,native
from umdiff.engine import Method
from umdiff.randomness import GlobalRandomStream
from umdiff.native_models import activate

STREAM=GlobalRandomStream()

def install(verify,nonzero_output,compare):
    globals().update(verify=verify,nonzero_output=nonzero_output,compare=compare)
    return {k[4:]:v for k,v in globals().items() if k.startswith('run_')}

def run_ppflow(steps):
    activate('ppflow',['models','transport','modules'])
    from transport.transport import Transport,ModelType,PathType,WeightType,Sampler
    record={'variants':[]}
    for levels in (2,3):
        model=nonzero_output(build('model',{'name':'native_ppflow','args':{'input_size':16,'patch_size':2,'patch_level':levels,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':0.,'learn_sigma':True}}))
        images=torch.randn(8,4,16,16);labels=torch.arange(8)
        objective=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,loss_type=WeightType.NONE,train_eps=0.,sample_eps=0.)
        method=Method({'name':'ppflow','batch':{'name':'ppflow','args':{'levels':levels}}})
        def original(net,step):return objective.training_losses(net,images,levels,{'y':labels})['loss']
        def port(net,step):return method.training(net,images,None,STREAM,model_kwargs={'y':labels}).loss
        a,b,item=verify(f'ppflow/{levels}',model,original,port,steps,clip=1.)
        initial=torch.randn(2,4,16,16);y=torch.tensor([0,1]);a.eval();b.eval()
        trajectory=Sampler(objective).sample_ode(sampling_method='euler',num_steps=13)(initial,a.forward_wo_cfg,y=y)
        # Native solver holds full-resolution state and invokes the stage-aware
        # backbone; only patch layout changes, with no information loss.
        state=initial.clone();grid=torch.linspace(0,1,13)
        for i,(t0,t1) in enumerate(zip(grid[:-1],grid[1:])):
            state=state+(t1-t0)*b.forward_wo_cfg(state,torch.ones(len(state))*t0,y)
            compare(f'ppflow/{levels}/trained_sampling_step{i}',state,trajectory[i+1],atol=2e-5,rtol=2e-4,scope='native_trained_model_inference_pixels')
        item['sampling']='13 endpoints, 12 Euler updates, same native stage-aware backbone; independent Euler arithmetic.'
        record['variants'].append(item)
    record['native_compatibility']='Released learn_sigma=False cannot unpatchify its output; preserve default True, which splits and discards variance channels for flow.'
    return record

def run_pixelflow(steps):
    activate('pixelflow',['pixelflow'])
    from pixelflow.data_in1k import collate_fn
    import pixelflow.data_in1k as data_module
    from diffusers.models.embeddings import _get_2d_rotary_pos_embed_np
    data_module.get_2d_rotary_pos_embed=_get_2d_rotary_pos_embed_np
    from pixelflow.scheduling_pixelflow import PixelFlowScheduler
    spec={'name':'adapter','args':{'time_keyword':'timestep','model':{'name':'native_pixelflow','args':{'in_channels':3,'out_channels':3,'num_attention_heads':2,'attention_head_dim':16,'depth':2,'patch_size':2,'num_classes':1000}}}}
    model=nonzero_output(build('model',spec));images=torch.randn(8,3,32,32);labels=torch.arange(8)
    config=SimpleNamespace(model=SimpleNamespace(params=SimpleNamespace(patch_size=2,attention_head_dim=16)),scheduler=SimpleNamespace(num_stages=3,num_train_timesteps=1000))
    scheduler=PixelFlowScheduler(1000,3)
    method=Method({'name':'pixelflow','batch':{'name':'pixelflow','args':{'stages':3}}})
    def original(net,step):
        batch=collate_fn(list(zip(images,labels.tolist())),config,scheduler)
        pred=net(batch['pixel_values'],batch['timesteps'],class_labels=torch.tensor(batch['input_ids']),latent_size=batch['batch_latent_size'],pos_embed=batch['pos_embed'],cu_seqlens_q=batch['cumsum_q_len'],cu_seqlens_k=None,seqlen_list_q=batch['seqlen_list_q'],seqlen_list_k=None)
        env=native.statements('pixelflow/train.py',120,126,dict(model_output=pred,target=batch['target_values'],batch=batch,torch=torch))
        return env['loss']
    def port(net,step):
        import random
        indices=(list(range(3))*(len(images)//3+1))[:len(images)];random.shuffle(indices)
        indices=torch.tensor(indices);t=torch.randint(0,1000,(len(images),))
        return method.training(net,images,t,STREAM,context={'stage_indices':indices,'class_labels':labels}).loss
    a,b,record=verify('pixelflow',model,original,port,steps,clip=1.)
    from pixelflow.pipeline_pixelflow import PixelFlowPipeline
    import pixelflow.pipeline_pixelflow as pipeline_module
    from umdiff.samplers import PixelFlowSampler
    pipeline_module.get_2d_rotary_pos_embed=_get_2d_rotary_pos_embed_np
    for cfg_scale in (0.,4.):
        native_pipe=PixelFlowPipeline(PixelFlowScheduler(1000,3),a.eval().model)
        torch.manual_seed(8091);expected=native_pipe([0,1],32,32,num_inference_steps=[4,4,4],guidance_scale=cfg_scale,device=torch.device('cpu'))
        torch.manual_seed(8091);initial=torch.randn(2,3,8,8)
        result=PixelFlowSampler(stages=3,steps=4,guidance_scale=cfg_scale)(b.eval(),initial,STREAM,{'class_labels':torch.tensor([0,1])})
        pixels=(result/2+.5).clamp(0,1).cpu().permute(0,2,3,1)
        compare(f'pixelflow/trained_model_pixels/cfg{cfg_scale}',pixels,expected,atol=3e-6,rtol=2e-5,scope='native_trained_model_inference_pixels')
    record['native_compatibility']='CPU fp32; packed FlashAttention replaced by per-sequence PyTorch SDPA on BOTH paths. Old default RoPE calls the retained NumPy-grid implementation directly, avoiding new Diffusers deprecated-API exception. GPU bf16 FlashAttention rounding is not certified.'
    return record

def run_dimr(steps):
    from umdiff.multiscale import dimr_targets
    def cfg(block,dim,size,ratio):return dict(block_type=block,dim=dim,hidden_dim=2*dim,num_attention_heads=4,num_blocks=1,max_height=size,max_width=size,image_input_ratio=1,input_feature_ratio=ratio,final_kernel_size=3,kernel_size=3,dropout_prob=0)
    config=dict(channels=2,num_classes=1001,block_grad_to_lowres=False,norm_type='TDRMSN',stage_configs=[cfg('TransformerBlock',32,8,2),cfg('ConvNeXtBlock',16,16,1)])
    spec={'name':'adapter','args':{'time_keyword':'log_snr','model':{'name':'native_dimr','args':{'config':config}}}}
    model=nonzero_output(build('model',spec));images=torch.randn(3,2,16,16);labels=torch.tensor([0,1,2])
    author=native.extract('dimr/train.py',['check_zip','get_skip','stp','mos','Schedule','LSimple'],dict(np=np,torch=torch,multi_scale_targets=native.load('dimr/sde.py').multi_scale_targets))
    betas=np.linspace(.00085**.5,.012**.5,16)**2;schedule=author['Schedule'](betas)
    method=Method({'name':'dimr','batch':{'name':'dimr','args':{'betas':betas.tolist()}}})
    def original(net,step):
        def call(state,log_snr,**kwargs):return net(state,log_snr,**kwargs)
        return author['LSimple'](images,call,schedule,[.25,1.],y=labels)
    def port(net,step):
        t=torch.tensor(np.random.choice(list(range(1,17)),(len(images),)))
        return method.training(net,images,t,STREAM,model_kwargs={'y':labels}).loss
    a,b,record=verify('dimr',model,original,port,steps)
    from umdiff.samplers import DiMRDPMSampler
    source=native.load('dimr/dpm_solver.py')
    schedule_noise=source.NoiseScheduleVP(schedule='discrete',betas=torch.tensor(betas).float())
    def model_fn(x,t):return a.eval()(x,(t*16).float()/1000*8-4,y=labels)[-1]
    initial=torch.randn_like(images)
    expected=source.DPM_Solver(model_fn,schedule_noise,predict_x0=True,thresholding=False).sample(initial,steps=12,eps=1/16,T=1.)
    actual=DiMRDPMSampler(betas.tolist(),12)(b.eval(),initial,model_kwargs={'y':labels})
    compare('dimr/trained_model_final_latents',actual,expected,atol=2e-5,rtol=2e-4,scope='native_trained_model_inference_latents')
    record['sampler_reuse']='Official DPM solver reused, identical released model time conversion; no VAE weights downloaded, so final object is a latent.'
    return record

def run_mdm(steps):
    model=nonzero_output(build('model',{'name':'native_mdm'}))
    from ml_mdm import diffusion as author,samplers
    cfg=author.NestedDiffusionConfig(sampler_config=samplers.SamplerConfig(num_diffusion_steps=16,prediction_type=samplers.PredictionType.V_PREDICTION,loss_target_type=samplers.PredictionType.DDPM,schedule_type=samplers.ScheduleType.DEEPFLOYD,schedule_shifted=True),use_double_loss=True,no_use_residual=True,use_vdm_loss_weights=False)
    native_pipe=author.NestedDiffusion(model,cfg)
    gammas=native_pipe.sampler.gammas.cpu().tolist()
    method=Method({'name':'mdm','batch':{'name':'mdm','args':{'gammas':gammas}},'prediction_adapter':{'name':'mdm_loss_coordinates'}})
    images=torch.randn(3,4,16,16);text=torch.randn(3,5,8);mask=torch.ones(3,5)
    def original(net,step):
        pipe=author.NestedDiffusion(net,cfg)
        return pipe.get_loss({'images':images,'lm_outputs':text,'lm_mask':mask})[0]
    def port(net,step):
        t=torch.randint(0,16,(len(images),))
        return method.training(net,images,t+1,STREAM,model_kwargs={'conditioning':text,'cond_mask':mask,'micros':{}}).loss
    a,b,record=verify('mdm',model,original,port,steps)
    from umdiff.samplers import MDMSampler
    initial=torch.randn(3,4,16,16);native_pipe=author.NestedDiffusion(a.eval(),cfg)
    torch.manual_seed(8093);state=initial.clone();expected_trace=[]
    for i in range(16,0,-1):
        state=native_pipe.sampler.get_xt_minus_1(native_pipe.model,i,state,text,mask)
        expected_trace.append(state)
    torch.manual_seed(8093);actual,actual_trace=MDMSampler(gammas)(b.eval(),initial,STREAM,{'conditioning':text,'cond_mask':mask,'micros':{}},True)
    for i,(x,y) in enumerate(zip(actual_trace[1:],expected_trace)):compare(f'mdm/trained_sampling_step{i}',x,y,atol=2e-5,rtol=2e-4,scope='native_trained_model_inference_pixels')
    record['native_compatibility']='Released 256 configuration: v prediction, epsilon loss, independent low-res noise, shifted schedule, no residual. Disabled VDM weights here are explicit controlled loss weighting.'
    return record

def run_ssd(steps):
    activate('ssd',['guided_diffusion'])
    from guided_diffusion import scale_space_diffusion as author
    from umdiff.ssd import ScaleSpaceSampler
    author.dist=SimpleNamespace(get_rank=lambda:0,barrier=lambda:None)
    mapping={8:8,16:16};betas=np.linspace(.001,.02,8)
    source=author.ScaleSpaceDiffusion(betas=betas,model_mean_type=author.ModelMeanType.START_X_RS,model_var_type=author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,ssd_config={'resolution_schedule':'equal','t_res_sampling_mode':'one_t_one_res','multires_training_mode':'flexi_unet','unet_res_layer_map':mapping,'scale_min':8},resolutions_array=[8,16],mse_loss_weight_type='constant')
    for op in source.ops.values():op.H_out,op.W_out=op.size_out
    model=nonzero_output(build('model',{'name':'native_ssd','args':{'image_size':16,'in_channels':3,'model_channels':32,'out_channels':3,'num_res_blocks':1,'attention_resolutions':[2],'channel_mult':[1,2],'unet_res_layer_map':mapping}}))
    method=Method({'name':'ssd','batch':{'name':'ssd','args':{'resolutions':[source.get_resolution(i) for i in range(8)],'betas':betas.tolist(),'unet_res_layer_map':mapping}},'prediction_adapter':{'name':'ssd_output'}})
    images=torch.randn(3,3,16,16)
    def original(net,step):return source.training_losses(net,images,torch.full((3,),step%8,dtype=torch.long),None)['loss']
    def port(net,step):return method.training(net,images,torch.full((3,),step%8,dtype=torch.long),STREAM).loss
    a,b,record=verify('ssd',model,original,port,steps)
    initial=torch.randn(3,3,8,8)
    torch.manual_seed(9901);expected=source.p_sample_loop(a.eval(),(3,3,16,16),noise=initial,device=torch.device('cpu'),progress=False)
    torch.manual_seed(9901);actual=ScaleSpaceSampler([source.get_resolution(i) for i in range(8)],betas.tolist(),mapping)(b.eval(),initial,STREAM)
    compare('ssd/trained_model_final_pixels',actual,expected,atol=5e-5,rtol=3e-4,scope='native_trained_model_inference_pixels')
    record['native_compatibility']='Missing BilinearOp output dimensions assigned on fixture objects; singleton barriers disabled; native t=0 residual-noise behavior preserved.'
    return record

def run_rdd(steps):
    import sys,hashlib
    sys.path.insert(0,str(native.ROOT.parent))
    from diffusion import gaussian_diffusion_with_drift as author
    from umdiff.gaussian_objective import GaussianSampler
    betas=np.linspace(.001,.02,16);images=torch.randn(3,4,16,16);labels=torch.tensor([0,1,2]);record={'variants':[]}
    for drift,learned,decoupled in ((False,False,False),(False,True,False),(True,False,False),(True,True,False),(True,False,True),(True,True,True)):
        kwargs={'c':0,'T_c':0,'T_c_plus_1':16,'timestep_map':list(range(16))} if drift else {}
        objective=author.GaussianDiffusion(betas=betas,model_mean_type=author.ModelMeanType.EPSILON,model_var_type=author.ModelVarType.LEARNED_RANGE if learned else author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,**kwargs)
        objective.timestep_map=list(range(16))
        path={'betas':betas.tolist(),'drift_start':0 if drift else None,'drift_end':16 if drift else None}
        model=nonzero_output(build('model',{'name':'native_rdd','args':{'input_size':16,'patch_size':2,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':0.,'learn_sigma':learned,'d_prediction':decoupled}}))
        method=Method({'name':'rdd_ddpm','batch':{'name':'rdd_ddpm','args':{'path':path,'learned_variance':learned,'decoupled':decoupled}}})
        def original(net,step):
            t=torch.tensor([0,1+(step%7),15]);fn=objective.training_losses_decouple if decoupled else objective.training_losses
            return fn(net,images,t,{'y':labels})['loss']
        def port(net,step):return method.training(net,images,torch.tensor([0,1+(step%7),15]),STREAM,model_kwargs={'y':labels}).loss
        name=f'rdd/drift{drift}/learned{learned}/decoupled{decoupled}'
        a,b,item=verify(name,model,original,port,steps)
        initial=torch.randn_like(images)
        torch.manual_seed(9501);expected=objective.p_sample_loop(a.eval(),initial.shape,noise=initial,device=torch.device('cpu'),model_kwargs={'y':labels},progress=False)
        if isinstance(expected,dict):expected=expected['sample']
        torch.manual_seed(9501);actual=GaussianSampler(path,learned,decoupled)(b.eval(),initial,STREAM,{'y':labels})
        compare(name+'/trained_sampling_pixels',actual,expected,atol=3e-5,rtol=3e-4,scope='native_trained_model_inference_latents')
        item.update(drift=drift,learned_variance=learned,decoupled=decoupled);record['variants'].append(item)
    for relative in ('diffusion/gaussian_diffusion_with_drift.py','diffusion/diffusion_utils.py','DiT/models.py'):
        file=native.ROOT.parent/relative
        native.AUDIT.append({'source':str(file),'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'execution':'existing migrated RDD source import, actual objective/backbone'})
    return record

def run_diffusability(steps):
    import einops,math
    from typing import Optional
    from umdiff.codec_training import CodecMethod
    class EasyDict(dict):
        __getattr__=dict.__getitem__
        __setattr__=dict.__setitem__
        @staticmethod
        def init_recursively(value):return EasyDict(value)
    vae_spec={'name':'autoencoder_kl','args':{'block_out_channels':[32,32],'down_block_types':['DownEncoderBlock2D']*2,'up_block_types':['UpDecoderBlock2D']*2,'latent_channels':4,'norm_num_groups':32,'layers_per_block':1}}
    core=build('model',vae_spec)
    vae_env=native.extract('diffusability/src/utils/autoencoder_utils.py',['vae_latents_forward','compute_kl'],dict(torch=torch,math=math,F=F,Optional=Optional,EasyDict=EasyDict))
    cfg=SimpleNamespace(arch='sd2_ae',model_variant='fp32',model_path=None,subfolder=None,pretrained=False,config_overrides=None,pretrained_lenient=False,pretrained_lenient_resume_kwargs=None,use_grad_checkpointing=False,freeze_encoder=False,extra_upsample=False,enc_split_input_size=None,dec_split_latent_size=None,is_temporal=False,use_mean_latents_in_eval=True)
    env=native.extract('diffusability/src/training/diffusers_ae.py',['DiffusersAutoEncoderBackbone'],dict(torch=torch,math=math,einops=einops,Optional=Optional,EasyDict=EasyDict,TensorGroup=object,init_diffusers_ae=lambda *a,**k:(core,4,(1,2,2),None),vae_latents_forward=vae_env['vae_latents_forward']))
    model=env['DiffusersAutoEncoderBackbone'](cfg,(1,16,16),3,3)
    # Execute actual released reconstruction/regularizer methods. No LPIPS
    # model is constructed; all its configured weights are zero.
    methods={n:native.method('diffusability/src/training/loss.py',n,dict(torch=torch,np=np,einops=einops,F=F,EasyDict=EasyDict,Optional=Optional,misc=SimpleNamespace(filter_nones=lambda d:EasyDict({k:v for k,v in d.items() if v is not None})),maybe_filter_loss_by_mask=lambda l,m:l,compute_img_grad_loss_per_frame=None,compute_video_freq2d_loss=None,compute_video_freq3d_loss=None)) for n in ('compute_rec_losses','maybe_compute_scale_equiv_reg')}
    images=torch.randn(2,3,16,16);record={'variants':[]}
    for mode in ('mean','bil'):
        settings=SimpleNamespace(**{n:SimpleNamespace(weight=1. if n=='mse_loss' else 0.,breadth_coef=.01) for n in ('perceptual_loss','perceptual_loss_alex','perceptual_loss_vgg','perceptual_loss_squeeze','mse_loss','mae_loss','pseudo_huber_loss','img_grad_loss','freq2d_loss','freq3d_loss','random_conv_l2_loss','video_random_conv_l2_loss')},scale_equiv_reg=SimpleNamespace(weight=.3,scale_factors=[.5],resample_strategy=mode,temporal_scale_factors=[],ignore_lpips=True,cut_dct2d_high_freqs=SimpleNamespace(max_cut_ratio=0.)))
        subject=SimpleNamespace(cfg=settings,lpips_pyr=None,lpips_alex=None,lpips_vgg=None,lpips_squeeze=None)
        subject.compute_rec_losses=lambda *a,**k:methods['compute_rec_losses'](subject,*a,**k)
        method=CodecMethod({'name':'diffusability','codec':{'reconstruction_style':'native','mse_weight':1.,'kl_weight':.0001,'regularizer_ignore_perceptual':False,'regularizers':[{'name':'scale_equivariance','args':{'factor':2,'mode':mode,'weight':.3}}]}})
        captured={}
        def original(net,step):
            prediction,ctx=net(images[:,None])
            captured.update(prediction=prediction[:,0].detach(),latents=ctx.latents[:,0].detach(),kl=ctx.kl.detach(),mean=ctx.latents_dist_mean[:,0].detach(),logvar=ctx.latents_dist_logvar[:,0].detach())
            loss=subject.compute_rec_losses(prediction,images[:,None]).rec+.0001*ctx.kl
            def call(a,b,c,*,latents,**kwargs):return net(None,encode=False,latents=latents)[0]
            result=methods['maybe_compute_scale_equiv_reg'](subject,call,EasyDict(total=loss),ctx.latents,images[:,None],None,None)
            return result.total
        def port(net,step):
            # This extra encoder is a diagnostic, not part of either loss graph.
            # Exclude only this pass from module-call alignment during tracing.
            from contextlib import nullcontext
            audit=verify.__globals__.get('CURRENT_AUDIT')
            with audit.suspended() if audit else nullcontext():
                posterior=net.model.encode(images).latent_dist
                compare('diffusability/'+mode+'/mean',posterior.mean,captured['mean'],atol=0,rtol=0)
                compare('diffusability/'+mode+'/logvar',posterior.logvar,captured['logvar'],atol=0,rtol=0)
            result=method.training(net,images,None,STREAM)
            compare('diffusability/'+mode+'/latent_draw',result.state,captured['latents'],atol=0,rtol=0)
            compare('diffusability/'+mode+'/reconstruction',result.prediction,captured['prediction'],atol=0,rtol=0)
            return result.loss
        a,b,item=verify('diffusability/'+mode,model,original,port,steps)
        latents=torch.randn(2,4,8,8)
        expected=a.eval()(None,encode=False,latents=latents[:,None])[0][:,0]
        actual=method.sample(b.eval(),latents)
        compare('diffusability/'+mode+'/trained_decoder_pixels',actual,expected,atol=2e-5,rtol=2e-4,scope='native_trained_model_inference_pixels')
        record['variants'].append(item)
    record['native_compatibility']='Actual DiffusersAutoEncoderBackbone forward, VAE sampling/KL, reconstruction and scale-equivariance methods executed. Asset factory returns local random AutoencoderKL. Native reconstruction returns Bx1 and adds B-vector KL, broadcasting BxB; native reconstruction_style preserves the same broadcast shape and SE addition order. Pixel-MSE branch only, no pretrained perceptual/GAN models or author long-run recipe claimed.'
    return record
