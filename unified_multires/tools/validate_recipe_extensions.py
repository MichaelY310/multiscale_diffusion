#!/usr/bin/env python3
"""Previously uncovered native branches. CPU, synthetic, no asset downloads."""
import argparse,json,os,sys,time,traceback,inspect,math,random
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import torch.nn.functional as F
import validate_training as training
import validate_training_more as more
import conformance
from activation_trace import ActivationAudit
from umdiff import build,native
from umdiff.engine import Method
from umdiff.randomness import GlobalRandomStream,RandomStream
from umdiff.native_models import activate
STREAM=GlobalRandomStream();CHECK=conformance.compare
FULL_CODEC=False
OUT=ROOT/'evidence/recipe_extensions'

def dctdiff(steps):
    source=native.load('dctdiff/sde.py')
    from umdiff.dctcodec import frequency_weights
    statistics={'Y_std':[6.522,3.377,3.386,2.389],'Cb_std':[6.1,2.4,2.1,1.8],'Cr_std':[6.,2.6,2.5,1.6],'block':2,'frequencies':4}
    weight=frequency_weights(**statistics)
    # Unchanged train.py normalization statements, independent of the port.
    env=native.statements('dctdiff/train.py',78,97,dict(config=SimpleNamespace(dataset=SimpleNamespace(**statistics,block_sz=2,low_freqs=4)),low2high_order=np.array([0,1,2,3]),logging=SimpleNamespace(info=lambda *a:None),np=np,torch=torch,device='cpu'))
    CHECK('dctdiff/EBFR_weight',weight,env['reweight_by_std'],atol=0,rtol=0)
    model=training.nonzero_output(build('model',{'name':'native_dctdiff','args':{'embed_dim':32,'depth':2,'num_heads':4,'tokens':16,'low_freqs':4,'num_classes':-1}}));clean=torch.randn(3,16,24)
    method=Method({'path':{'name':'vp','args':{'snr_scale':4.}},'target':{'name':'joint','args':{'quantity':'epsilon'}},'clock':{'name':'linear','args':{'network_scale':999.}},'metric':{'name':'mse','args':{'dct_statistics':statistics}}})
    def original(net,step):return source.LSimple(source.ScoreModel(net,'noise_pred',source.VPSDE(SNR_scale=4.)),clean,reweight=env['reweight_by_std'])
    def port(net,step):return method.training(net,clean,torch.rand(3),STREAM).loss
    a,b,record=training.verify('dctdiff_EBFR',model,original,port,steps)
    from umdiff.samplers import DCTDPMSampler
    initial=torch.randn_like(clean);CHECK('dctdiff/weighted_trained_DPM',DCTDPMSampler(snr_scale=4.)(b.eval(),initial),DCTDPMSampler(snr_scale=4.)(a.eval(),initial),atol=0,rtol=0)
    return record

def mdm(steps,vdm=False):
    model=training.nonzero_output(build('model',{'name':'native_mdm'}))
    from ml_mdm import diffusion as author,samplers
    cfg=author.NestedDiffusionConfig(sampler_config=samplers.SamplerConfig(num_diffusion_steps=16,prediction_type=samplers.PredictionType.V_PREDICTION,loss_target_type=samplers.PredictionType.DDPM,schedule_type=samplers.ScheduleType.DEEPFLOYD,schedule_shifted=True),use_double_loss=True,no_use_residual=True,use_vdm_loss_weights=False,mixed_ratio='2:1')
    cfg.use_vdm_loss_weights=vdm
    pipe=author.NestedDiffusion(model,cfg);gammas=pipe.sampler.gammas.cpu().tolist()
    if vdm:CHECK('mdm/native_VDM_weights',build('time_sampler',{'name':'discrete','args':{'steps':16,'offset':1,'vdm_gammas':gammas}}).weights,pipe.sampler.vdm_loss_weights,atol=0,rtol=0)
    method=Method({'batch':{'name':'mdm','args':{'gammas':gammas,'mixed_ratio':'2:1'}},'prediction_adapter':{'name':'mdm_loss_coordinates'}})
    images=torch.randn(3,4,16,16);text=torch.randn(3,5,8);mask=torch.ones(3,5)
    def original(net,step):
        result=author.NestedDiffusion(net,cfg).get_loss({'images':images,'lm_outputs':text,'lm_mask':mask})
        return (result[0]*result[-1]).sum()/result[-1].sum() if vdm else result[0]
    def port(net,step):
        t,context=build('time_sampler',{'name':'discrete','args':{'steps':16,'offset':1,**({'vdm_gammas':gammas} if vdm else {})}})(len(images),'cpu',STREAM)
        loss=method.training(net,images,t,STREAM,model_kwargs={'conditioning':text,'cond_mask':mask,'micros':{}}).loss
        return (loss*context['loss_weights']).sum()/context['loss_weights'].sum() if vdm else loss
    a,b,record=training.verify('mdm_mixed_2to1'+('_vdm' if vdm else ''),model,original,port,steps)
    from umdiff.guidance import threshold_sample
    from ml_mdm.samplers import ThresholdType
    subject=pipe.sampler
    for kind,enum in [('clip',ThresholdType.CLIP),('dynamic',ThresholdType.DYNAMIC),('dynamic_if',ThresholdType.DYNAMIC_IF)]:
        subject._config.threshold_function=enum;x=torch.randn(3,4,16,16)*5
        for scale in (1.,2.):CHECK('mdm/threshold/'+kind+str(scale),threshold_sample(x,kind,scale),subject.clip_sample(x,scale),atol=0,rtol=0)
    # Author negative-first batched text guidance.
    from umdiff.samplers import MDMSampler
    initial=torch.randn(3,4,16,16);negative=torch.randn_like(text);negative_mask=torch.ones_like(mask)
    cfg.mixed_ratio=None;cfg.sampler_config.threshold_function=ThresholdType.DYNAMIC
    native_pipe=author.NestedDiffusion(a.eval(),cfg)
    both_text=torch.cat([negative,text]);both_mask=torch.cat([negative_mask,mask]);state=initial.clone();expected=[]
    training.seed(9081)
    for i in range(16,0,-1):
        state=native_pipe.sampler.get_xt_minus_1(native_pipe.model,i,state,both_text,both_mask,guidance_scale=3.)
        expected.append(state)
    training.seed(9081)
    value,trace=MDMSampler(gammas,guidance_scale=3.,threshold='dynamic')(b.eval(),initial,STREAM,{'conditioning':text,'cond_mask':mask,'micros':{},'unconditional':{'conditioning':negative,'cond_mask':negative_mask,'micros':{}}},True)
    for i,(p,q) in enumerate(zip(trace[1:],expected)):CHECK(f'mdm/guided_dynamic/step{i}',p,q,atol=0,rtol=0)
    return record

def ssd(steps):
    activate('ssd',['guided_diffusion'])
    from guided_diffusion import scale_space_diffusion as author,resample
    author.dist=SimpleNamespace(get_rank=lambda:0,barrier=lambda:None)
    mapping={8:8,16:16};betas=np.linspace(.001,.02,8)
    settings={'resolution_schedule':'equal','t_res_sampling_mode':'one_res_many_t_except_one_t_for_res_change','multires_training_mode':'flexi_unet','unet_res_layer_map':mapping,'scale_min':8}
    source=author.ScaleSpaceDiffusion(betas=betas,model_mean_type=author.ModelMeanType.START_X_RS,model_var_type=author.ModelVarType.FIXED_SMALL,loss_type=author.LossType.MSE,ssd_config=settings,resolutions_array=[8,16],mse_loss_weight_type='min_snr_5',rescale_timesteps=True)
    for op in source.ops.values():op.H_out,op.W_out=op.size_out
    resolutions=[source.get_resolution(i) for i in range(8)]
    model=training.nonzero_output(build('model',{'name':'native_ssd','args':{'image_size':16,'in_channels':3,'model_channels':32,'out_channels':3,'num_res_blocks':1,'attention_resolutions':[2],'channel_mult':[1,2],'unet_res_layer_map':mapping}}))
    method=Method({'batch':{'name':'ssd','args':{'resolutions':resolutions,'betas':betas.tolist(),'unet_res_layer_map':mapping,'loss_weight':'min_snr_5','rescale_timesteps':True}},'prediction_adapter':{'name':'ssd_output'}})
    from umdiff.time_sampling import SSDTimes
    sampler=SSDTimes(resolutions);native_sampler=resample.UniformSampler(source);native_sampler.prepare_sampling(settings,[8,16]);images=torch.randn(3,3,16,16);captured={}
    def original(net,step):
        t,weight,_=native_sampler.sample(3,'cpu',settings,[8,16]);captured.update(time=t,weight=weight)
        return source.training_losses(net,images,t,None)['loss']*weight
    def port(net,step):
        t,ctx=sampler(3,'cpu',STREAM);CHECK('ssd/time_group',t,captured['time'],atol=0,rtol=0);CHECK('ssd/importance',ctx['importance_weights'],captured['weight'],atol=0,rtol=0)
        return method.training(net,images,t,STREAM,context=ctx).loss
    a,b,record=training.verify('ssd_mixed_min_snr_rescaled',model,original,port,steps)
    from umdiff.ssd import ScaleSpaceSampler
    initial=torch.randn(3,3,8,8);training.seed(9901);expected=source.p_sample_loop(a.eval(),(3,3,16,16),noise=initial,device=torch.device('cpu'),progress=False)
    training.seed(9901);actual=ScaleSpaceSampler(resolutions,betas.tolist(),mapping,rescale_timesteps=True)(b.eval(),initial,STREAM)
    CHECK('ssd/rescaled_sampling',actual,expected,atol=0,rtol=0)
    return record

def ppflow(steps):
    activate('ppflow',['models','transport','modules'])
    from transport.transport import Transport,ModelType,PathType,WeightType,Sampler
    from umdiff.samplers import PPFlowSampler
    from umdiff.recipes import initialize
    records=[]
    for levels in (2,3):
        model=training.nonzero_output(build('model',{'name':'native_ppflow','args':{'input_size':16,'patch_size':2,'patch_level':levels,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':.1,'learn_sigma':True}}))
        # Source-shaped SiT checkpoint: shared layers and the unexpanded heads.
        state={k:v.clone() for k,v in model.state_dict().items() if not any(s in k for s in ('proj1','proj2','linear1','linear2'))}
        file=OUT/f'ppflow_sit_shape_{levels}.pt';torch.save(state,file)
        reference=deepcopy(model);initialize(model,{'path':str(file),'mode':'ppflow_sit'})
        weight_helpers=native.extract('ppflow/train.py',['initialize_layer_weights','initialize_layer_bias'],dict(torch=torch))
        reference.load_state_dict(state,strict=False)
        for suffix,repeat in [('1',4)]+([('2',2)] if levels==3 else []):
            weight_helpers['initialize_layer_weights'](state['x_embedder.proj.weight'],getattr(reference.x_embedder,'proj'+suffix),repeat,float(repeat),1)
            weight_helpers['initialize_layer_bias'](state['x_embedder.proj.bias'],getattr(reference.x_embedder,'proj'+suffix))
            weight_helpers['initialize_layer_weights'](state['final_layer.linear.weight'],getattr(reference.final_layer,'linear'+suffix),repeat,dim=0)
            weight_helpers['initialize_layer_bias'](state['final_layer.linear.bias'],getattr(reference.final_layer,'linear'+suffix),repeat,0)
        for key,p in model.state_dict().items():CHECK('ppflow/init/'+key,p,reference.state_dict()[key],atol=0,rtol=0)
        images=torch.randn(8,4,16,16);labels=torch.arange(8);transport=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,loss_type=WeightType.NONE,train_eps=0.,sample_eps=0.)
        method=Method({'batch':{'name':'ppflow','args':{'levels':levels}}})
        def original(net,step):return transport.training_losses(net,images,levels,{'y':labels})['loss']
        def port(net,step):return method.training(net,images,None,STREAM,model_kwargs={'y':labels}).loss
        a,b,record=training.verify('ppflow_init_dropout_'+str(levels),model,original,port,steps)
        initial=torch.randn(2,4,16,16);y=torch.tensor([0,1]);scales=[1.4,2.1,3.2][:levels];a.eval();b.eval()
        expected=Sampler(transport).sample_ode(sampling_method='euler',num_steps=13)(torch.cat([initial,initial]),a.forward_with_cfg,y=torch.cat([y,torch.full_like(y,1000)]),cfg_scales=scales)
        value,trace=PPFlowSampler(cfg_scales=scales)(b,initial,model_kwargs={'y':y},return_trajectory=True)
        for i,(p,q) in enumerate(zip(trace,expected)):CHECK(f'ppflow/{levels}/guided_step{i}',p,q[:2],atol=0,rtol=0)
        records.append(record)
    return {'variants':records}

def dimr(steps):
    from umdiff.cli import conditions
    from torch.utils.data import Dataset
    def cfg(block,dim,size,ratio):return dict(block_type=block,dim=dim,hidden_dim=2*dim,num_attention_heads=4,num_blocks=1,max_height=size,max_width=size,image_input_ratio=1,input_feature_ratio=ratio,final_kernel_size=3,kernel_size=3,dropout_prob=0)
    config=dict(channels=2,num_classes=1001,block_grad_to_lowres=False,norm_type='TDRMSN',stage_configs=[cfg('TransformerBlock',32,8,2),cfg('ConvNeXtBlock',16,16,1)])
    spec={'name':'adapter','args':{'time_keyword':'log_snr','model':{'name':'native_dimr','args':{'config':config}}}}
    model=training.nonzero_output(build('model',spec));images=torch.randn(4,2,16,16);labels=torch.arange(4)
    cfgdataset=native.extract('dimr/datasets.py',['CFGDataset'],dict(Dataset=Dataset,random=random))['CFGDataset'](list(zip(images,labels.tolist())),.1,1000)
    dropout={'conditioning':{'y':{'kind':'labels','classes':1000,'dropout':.1,'dropout_rng':'python','null_label':1000}}}
    author=native.extract('dimr/train.py',['check_zip','get_skip','stp','mos','Schedule','LSimple'],dict(np=np,torch=torch,multi_scale_targets=native.load('dimr/sde.py').multi_scale_targets))
    betas=np.linspace(.00085**.5,.012**.5,16)**2;schedule=author['Schedule'](betas);method=Method({'batch':{'name':'dimr','args':{'betas':betas.tolist()}}});captured={}
    def original(net,step):
        y=torch.tensor([cfgdataset[i][1] for i in range(4)]);captured['y']=y
        return author['LSimple'](images,lambda x,log_snr,**kw:net(x,log_snr,**kw),schedule,[.25,1.],y=y)
    def port(net,step):
        kwargs,_=conditions(dropout,{'labels':labels},torch.device('cpu'),4,stream=STREAM);CHECK('dimr/python_dropout',kwargs['y'],captured['y'],atol=0,rtol=0)
        t=torch.tensor(np.random.choice(list(range(1,17)),(4,)))
        return method.training(net,images,t,STREAM,model_kwargs=kwargs).loss
    a,b,record=training.verify('dimr_python_dropout',model,original,port,steps)
    from umdiff.samplers import DiMRDPMSampler
    source=native.load('dimr/dpm_solver.py');noise_schedule=source.NoiseScheduleVP(schedule='discrete',betas=torch.tensor(betas).float());initial=torch.randn_like(images)
    for scale,power,scaling in ((.4,-1,-1),(2.,2.,-1),(2.,2.,2.)):
        env=native.statements('dimr/eval.py',93,115,dict(nnet=lambda x,log_snr,**kw:a.eval()(x,log_snr,**kw),dataset=SimpleNamespace(K=1000),device='cpu',cfg_scale=scale,cfg_scale_pow=power,torch=torch,math=math))
        env.update(N=16,kwargs={'y':labels});env=native.statements('dimr/eval.py',134,142,env)
        expected=source.DPM_Solver(env['model_fn'],noise_schedule,predict_x0=True,thresholding=False,scaling_factor=scaling).sample(initial,steps=12,eps=1/16,T=1.)
        actual=DiMRDPMSampler(betas.tolist(),12,guidance_scale=scale,guidance_power=power,scaling_factor=scaling)(b.eval(),initial,model_kwargs={'y':labels})
        CHECK(f'dimr/cfg{scale}_power{power}_scaling{scaling}',actual,expected,atol=0,rtol=0)
    return record

def dfm(steps):
    record=training.run_dfm(steps,dropout=.1)
    from dfm_utils.samplers_dfm import dfm_euler_sampler
    from umdiff.representations import Laplacian
    from umdiff.recipes import state_file
    spec={'name':'adapter','args':{'output_index':0,'model':{'name':'native_dfm','args':{'input_size':16,'in_channels':4,'hidden_size':32,'decoder_hidden_size':32,'encoder_depth':1,'depth':2,'num_heads':4,'class_dropout_prob':.1,'enable_repa':False,'qk_norm':False}}}}
    a=build('model',spec);b=build('model',spec);checkpoint=training.CHECKPOINT_ROOT/'dfm';a.load_state_dict(state_file(checkpoint/'native.pt','model'));b.load_state_dict(state_file(checkpoint/'unified.pt','model'))
    rep=Laplacian(2,up='bilinear',order='coarse_first');images=torch.randn(3,4,16,16);labels=torch.arange(3);initial=dict(enumerate(torch.randn_like(x) for x in rep.encode(images)))
    for window in ((0.,1.),(.2,.8),(.1,.4)):
        expected=dfm_euler_sampler(lambda x,t,**kw:(a.eval()(x,t,**kw),None),initial,labels,num_steps_per_scale=[4,4],stage_thresholds=[.5],cfg_scale=2.5,guidance_low=window[0],guidance_high=window[1])
        method=Method({'representation':{'name':'laplacian','args':{'levels':2,'up':'bilinear','order':'coarse_first'}},'clock':{'name':'dfm'},'guidance':{'name':'cfg_window','args':{'scale':2.5,'low':window[0],'high':window[1]}}})
        CHECK('dfm/guidance_window/'+str(window),method.sample(b.eval(),initial,model_kwargs={'y':labels}),rep.decode(expected),atol=0,rtol=0)
    return record

def lapflow(steps):
    activate('lapflow',['models','sampler'])
    from sampler import ours_sample as author
    from umdiff.samplers import LapFlowSampler
    from unittest.mock import patch
    spec={'name':'adapter','args':{'time_first':True,'model':{'name':'native_lapflow','args':{'input_size':[16,8],'patch_size':2,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':.1,'use_mot':True,'use_mask':True}}}}
    model=training.nonzero_output(build('model',spec));images=torch.randn(3,4,16,16);labels=torch.arange(3);records=[]
    for curve in ('linear','cosine','square'):
        settings=SimpleNamespace(noise_var_factor=1.3,path_type='square' if curve=='square' else 'linear',use_gvp=curve=='cosine',epsilon=None,variance_aware=True,use_mot=True,loss_weight_scale_0=1.,loss_weight_scale_1=1.,num_levels=2)
        method=Method({'batch':{'name':'lapflow','args':{'levels':2,'curve':curve,'noise_var_factor':1.3,'variance_aware':True}}})
        def original(net,step):
            stage=step%2;t=torch.tensor([.57,.71,.89]) if stage==0 else torch.tensor([.07,.21,.49]);u,xt,weights,var=author.generate_xt_and_u([.3,.6],.4,images,t[:,None,None,None],stage,2,False,settings)
            pred=net(xt,t,y=labels,var_scale=var)
            return sum(F.mse_loss(p.float(),q.float(),reduction='mean')*w for p,q,w in zip(pred,u,weights))
        def port(net,step):
            stage=step%2;t=torch.tensor([.57,.71,.89]) if stage==0 else torch.tensor([.07,.21,.49])
            return method.training(net,images,t,STREAM,context={'stage':stage},model_kwargs={'y':labels}).loss
        a,b,item=training.verify('lapflow_optional_'+curve,model,original,port,steps,clip=1.)
        if curve=='square':records.append(item);continue
        settings.__dict__.update(method='euler',step_size=.1,perturb=False,compute_nfe=False,cfg_scale=2.5,time_seg_small=.4,time_segments=[.3,.6],atol=1e-5,rtol=1e-5)
        initial=torch.randn(3,4,16,16);both=torch.cat([initial,initial]);y=torch.cat([labels,torch.full_like(labels,1000)]);old_tensor=torch.tensor
        def cpu_tensor(*args,**kw):
            if kw.get('device')=='cuda':kw['device']='cpu'
            return old_tensor(*args,**kw)
        with patch.object(author.torch,'tensor',cpu_tensor):expected=author.ours_sample_from_model([a.eval().model],0,both,{'y':y,'cfg_scale':2.5},settings,False,largest_only=True)[:3]
        actual=LapFlowSampler(curve=curve,guidance_scale=2.5,noise_var_factor=1.3,variance_aware=True)(b.eval(),initial,STREAM,{'y':labels})
        CHECK('lapflow/'+curve+'/guided_pixels',actual,expected,atol=2e-5,rtol=2e-4);records.append(item)
    return {'variants':records,'square_sampling':'Deferred: native source swaps square_get_noise_at_seg arguments; explicit error rather than silently changing native behavior.'}

def lapflow3(steps):
    activate('lapflow',['models','sampler'])
    from sampler import ours_sample as author
    from umdiff.samplers import LapFlowSampler
    from unittest.mock import patch
    spec={'name':'adapter','args':{'time_first':True,'model':{'name':'native_lapflow','args':{'input_size':[16,8,4],'patch_size':2,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':.1,'use_mot':True,'use_mask':True}}}}
    model=training.nonzero_output(build('model',spec));images=torch.randn(3,4,16,16);labels=torch.arange(3)
    settings=SimpleNamespace(noise_var_factor=None,path_type='linear',use_gvp=False,epsilon=None,variance_aware=False,use_mot=True,loss_weight_scale_0=1.,loss_weight_scale_1=1.,num_levels=3)
    method=Method({'batch':{'name':'lapflow','args':{'levels':3}}})
    def clock(step):
        stage=step%3;return stage,[torch.tensor([.67,.79,.91]),torch.tensor([.37,.61,.89]),torch.tensor([.07,.21,.49])][stage]
    def original(net,step):
        stage,t=clock(step);u,xt,weights,var=author.generate_xt_and_u([.3,.6],.4,images,t[:,None,None,None],stage,3,False,settings);pred=net(xt,t,y=labels,var_scale=var)
        return sum(F.mse_loss(p.float(),q.float(),reduction='mean')*w for p,q,w in zip(pred,u,weights))
    def port(net,step):
        stage,t=clock(step);return method.training(net,images,t,STREAM,context={'stage':stage},model_kwargs={'y':labels}).loss
    a,b,record=training.verify('lapflow_three_level_native',model,original,port,steps,clip=1.)
    settings.__dict__.update(method='euler',step_size=.1,perturb=False,compute_nfe=False,cfg_scale=2.5,time_seg_small=.4,time_segments=[.3,.6],atol=1e-5,rtol=1e-5)
    initial=torch.randn(3,4,16,16);both=torch.cat([initial,initial]);y=torch.cat([labels,torch.full_like(labels,1000)]);old_tensor=torch.tensor
    def cpu_tensor(*args,**kw):
        if kw.get('device')=='cuda':kw['device']='cpu'
        return old_tensor(*args,**kw)
    training.seed(5081)
    with patch.object(author.torch,'tensor',cpu_tensor):expected=author.ours_sample_from_model([a.eval().model],0,both,{'y':y,'cfg_scale':2.5},settings,False,largest_only=True)[:3]
    expected_rng=torch.get_rng_state().clone();training.seed(5081)
    actual=LapFlowSampler(levels=3,guidance_scale=2.5)(b.eval(),initial,STREAM,{'y':labels});CHECK('lapflow/three_level/guided_final',actual,expected,atol=0,rtol=0);CHECK('lapflow/three_level/sampler_RNG',torch.get_rng_state(),expected_rng,atol=0,rtol=0)
    return record

def relay(steps):
    record=training.run_relay(steps)
    import scipy.stats as stats
    from umdiff.relay import RelayChainSampler
    blur=native.load('relay/blurring.py');loss=native.extract('relay/training/loss.py',['EDMLoss'],dict(torch=torch,np=np,persistence=SimpleNamespace(persistent_class=lambda cls:cls)))['EDMLoss']()
    spec={'name':'native_relay','args':{'img_resolution':8,'img_channels':3,'model_channels':64,'channel_mult':[1,2],'num_blocks':1,'dropout':0.,'attn_resolutions':[4]}}
    model=training.nonzero_output(build('model',spec));images=torch.randn(2,3,8,8);method=Method({'batch':{'name':'edm'}})
    a,b,low=training.verify('relay_edm_low',model,lambda net,s:loss(net,images).flatten(1).mean(1),lambda net,s:method.training(net,images,None,STREAM).loss,steps)
    highspec={**spec,'args':{**spec['args'],'img_resolution':16,'attn_resolutions':[8]}};ha=build('model',highspec);hb=build('model',highspec)
    from umdiff.recipes import state_file
    ha.load_state_dict(state_file(training.CHECKPOINT_ROOT/'relay/native.pt','model'));hb.load_state_dict(state_file(training.CHECKPOINT_ROOT/'relay/unified.pt','model'))
    env=native.extract('relay/generate_imagenet.py',['edm_sampler','blur_sampler'],dict(torch=torch,np=np,block_noise=blur.block_noise,dct_2d=blur.dct_2d,idct_2d=blur.idct_2d,get_alpha_t=lambda t,patch,device,**kw:blur.get_alpha_t(t,patch,device,repeat_dims=16//patch,**kw)))
    initial=torch.randn(2,3,8,8);training.seed(7731);coarse=env['edm_sampler'](a.eval(),initial,num_steps=12,sigma_max=2.)
    expected=env['blur_sampler'](ha.eval(),F.interpolate(coarse,16),num_steps=12,sigma_max=2.,truncation_sigma=1.,up_scale=4,s_block=.15,s_noise=.2)
    training.seed(7731);actual=RelayChainSampler(output_size=16,first={'num_steps':12,'sigma_max':2.},second={'steps':12,'sigma_max':2.,'truncation_sigma':1.})(torch.nn.ModuleDict({'low':b.eval(),'high':hb.eval()}),initial,STREAM)
    CHECK('relay/full_chain_pixels',actual,expected,atol=2e-5,rtol=2e-4);record['low_stage']=low
    return record

def rdd(steps):
    from umdiff.rdd_native import diffusion_backend,NativeRDDStages
    from umdiff.gaussian_objective import GaussianSampler
    backend=diffusion_backend();model=training.nonzero_output(build('model',{'name':'native_rdd_unified','args':{'patch_size':2,'in_channels':4,'hidden_size':32,'depth':2,'num_heads':4,'class_dropout_prob':.1,'learn_sigma':True,'base_size':8}}));images=torch.randn(3,4,16,16);labels=torch.arange(3)
    resolutions=[16,8];limits=[0,8,16];diffusions=[backend.create_diffusion_with_drift('',diffusion_steps=16,noise_schedule='squaredcos_cap_v2',c=0 if i==0 else None,T_c=limits[i],T_c_plus_1=limits[i+1],sample_scale=2) for i in range(2)]
    stages=[]
    for i,diffusion in enumerate(diffusions):
        path={'betas':diffusion.betas.tolist(),'timestep_map':diffusion.timestep_map,'drift_start':limits[i] if i==0 else None,'drift_end':limits[i+1] if i==0 else None}
        stages.append({'input':{'name':'average_pool','args':{'factor':2**i,'detach':i>0}},'time_sampler':{'name':'discrete','args':{'steps':diffusion.num_timesteps}},'method':{'batch':{'name':'rdd_ddpm','args':{'path':path,'learned_variance':True}}}})
    method=Method({'training_stages':stages,'training_stage_reduction':'sum_stage_means'})
    def original(net,step):
        loss=0.
        for i,diffusion in enumerate(diffusions):
            x=images if i==0 else images.detach().reshape(3,4,8,2,8,2).mean((3,5));t=torch.randint(0,diffusion.num_timesteps,(3,));loss=loss+diffusion.training_losses(net,x,t,{'y':labels})['loss'].mean()
        return loss
    a,b,record=training.verify('rdd_multires_training_mapped_time',model,original,lambda net,s:method.training(net,images,None,STREAM,model_kwargs={'y':labels}).loss,steps)
    # Source stage dispatcher and source fresh-noise boundary, not our bridges.
    from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
    env=native.extract_file(native.ROOT.parent/'2d/sample_fid_unified.py',['_redraw_upsample','_run_stage'],dict(torch=torch,_extract_into_tensor=_extract_into_tensor)) if hasattr(native,'extract_file') else None
    if env is None:
        import ast
        file=native.ROOT.parent/'2d/sample_fid_unified.py';tree=ast.parse(file.read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('_redraw_upsample','_run_stage')];env=dict(torch=torch,_extract_into_tensor=_extract_into_tensor);exec(compile(ast.Module(body=selected,type_ignores=[]),str(file),'exec'),env)
    initial=torch.randn(3,4,8,8)
    for solver in ('jump','dpm1','dpm2'):
        sampler=NativeRDDStages(resolutions=resolutions,stage_lengths=[8,8],steps=[4,4],noise_schedule='squaredcos_cap_v2',solver=solver,guidance_scale=2.)
        kw={'y':torch.cat([labels,torch.full_like(labels,1000)]),'cfg_scale':2.};training.seed(7131);state=torch.cat([initial,torch.randn_like(initial)]);trace=[]
        for position,index in enumerate((1,0)):
            d=sampler.diffusions[index]
            if position:
                state,_=env['_redraw_upsample'](out['pred_xstart'][:3].float(),torch.full((3,),d.num_timesteps-1,dtype=torch.long),d,16);state=torch.cat([state,state]);trace.append(state[:3])
            out=env['_run_stage'](d,a.eval().forward_with_cfg,state,kw,'cpu','ddpm' if solver=='jump' else solver);state=out['sample'];trace.append(state[:3])
        training.seed(7131);actual,observed=sampler(b.eval(),initial,STREAM,{'y':labels},True)
        for i,(p,q) in enumerate(zip(observed,trace)):CHECK('rdd/'+solver+'/stage_boundary'+str(i),p,q,atol=0,rtol=0)
        CHECK('rdd/'+solver+'/final',actual,state[:3].float(),atol=0,rtol=0)
    return record

def diffusability(steps):
    if not FULL_CODEC:torch.set_num_threads(1)
    import einops,math
    from typing import Optional
    from umdiff.codec_training import CodecMethod
    from umdiff.regularizers import native_dct_utilities
    class EasyDict(dict):
        __getattr__=dict.__getitem__;__setattr__=dict.__setitem__
        @staticmethod
        def init_recursively(value):return EasyDict(value)
    channels=[128,256,512,512] if FULL_CODEC else [32,32]
    core=build('model',{'name':'autoencoder_kl','args':{'block_out_channels':channels,'down_block_types':['DownEncoderBlock2D']*len(channels),'up_block_types':['UpDecoderBlock2D']*len(channels),'latent_channels':4,'norm_num_groups':32,'layers_per_block':2 if FULL_CODEC else 1}})
    vae=native.extract('diffusability/src/utils/autoencoder_utils.py',['vae_latents_forward','compute_kl'],dict(torch=torch,math=math,F=F,Optional=Optional,EasyDict=EasyDict))
    cfg=SimpleNamespace(arch='sd2_ae',model_variant='fp32',model_path=None,subfolder=None,pretrained=False,config_overrides=None,pretrained_lenient=False,pretrained_lenient_resume_kwargs=None,use_grad_checkpointing=False,freeze_encoder=False,extra_upsample=False,enc_split_input_size=None,dec_split_latent_size=None,is_temporal=False,use_mean_latents_in_eval=True)
    compression=8 if FULL_CODEC else 2
    env=native.extract('diffusability/src/training/diffusers_ae.py',['DiffusersAutoEncoderBackbone'],dict(torch=torch,math=math,einops=einops,Optional=Optional,EasyDict=EasyDict,TensorGroup=object,init_diffusers_ae=lambda *a,**kw:(core,4,(1,compression,compression),None),vae_latents_forward=vae['vae_latents_forward']))
    resolution=256 if FULL_CODEC else 64
    model=env['DiffusersAutoEncoderBackbone'](cfg,(1,resolution,resolution),3,3)
    model.model.encoder.requires_grad_(False);model.model.quant_conv.requires_grad_(False)
    method=CodecMethod({'task':'codec','codec':{'reconstruction_style':'native','mse_weight':1.,'kl_weight':0.,'perceptual':{'weight':.1,'model':{'name':'native_perceptual_pyramid','args':{'synthetic_fixture':True,'scales':[1],'loss_weights':[1,1,1,1,1]}}},'regularizers':[{'name':'random_scale_equivariance','args':{'factors':[2,4],'weight':.3}}],'regularizer_ignore_perceptual':False}})
    if FULL_CODEC:
        method=CodecMethod({'task':'codec','codec':{'reconstruction_style':'native','mse_weight':0.,'kl_weight':0.,'perceptual':{'weight':1.,'model':{'name':'native_perceptual_pyramid','args':{'synthetic_fixture':True,'scales':[1.,.5,.25],'loss_weights':[1,1,1,1,1]}}},'regularizers':[{'name':'random_scale_equivariance','args':{'factors':[2,4],'weight':1.}}],'regularizer_ignore_perceptual':False}})
    settings=SimpleNamespace(**{n:SimpleNamespace(weight=.1 if n=='perceptual_loss' else 1. if n=='mse_loss' else 0.,breadth_coef=.01) for n in ('perceptual_loss','perceptual_loss_alex','perceptual_loss_vgg','perceptual_loss_squeeze','mse_loss','mae_loss','pseudo_huber_loss','img_grad_loss','freq2d_loss','freq3d_loss','random_conv_l2_loss','video_random_conv_l2_loss')},scale_equiv_reg=SimpleNamespace(weight=.3,scale_factors=[.5,.25],resample_strategy='mean',temporal_scale_factors=[],ignore_lpips=False,cut_dct2d_high_freqs=SimpleNamespace(max_cut_ratio=0.)))
    methods={n:native.method('diffusability/src/training/loss.py',n,dict(torch=torch,np=np,einops=einops,F=F,EasyDict=EasyDict,Optional=Optional,misc=SimpleNamespace(filter_nones=lambda d:EasyDict({k:v for k,v in d.items() if v is not None})),maybe_filter_loss_by_mask=lambda l,m:l,compute_img_grad_loss_per_frame=None,compute_video_freq2d_loss=None,compute_video_freq3d_loss=None)) for n in ('compute_rec_losses','maybe_compute_scale_equiv_reg')}
    if FULL_CODEC:
        settings.mse_loss.weight=0.;settings.perceptual_loss.weight=1.;settings.scale_equiv_reg.weight=1.
    subject=SimpleNamespace(cfg=settings,lpips_pyr=method.perceptual,lpips_alex=None,lpips_vgg=None,lpips_squeeze=None);subject.compute_rec_losses=lambda *a,**kw:methods['compute_rec_losses'](subject,*a,**kw);images=torch.randn(2,3,resolution,resolution)
    saved_factory=training.AUDIT_FACTORY
    if FULL_CODEC and saved_factory:
        training.AUDIT_FACTORY=lambda name,pairs:saved_factory(name,pairs+[('perceptual',method.perceptual,method.perceptual)])
    def original(net,step):
        prediction,ctx=net(images[:,None]);loss=subject.compute_rec_losses(prediction,images[:,None]).rec
        def call(a,b,c,*,latents,**kw):return net(None,encode=False,latents=latents)[0]
        return methods['maybe_compute_scale_equiv_reg'](subject,call,EasyDict(total=loss),ctx.latents,images[:,None],None,None).total
    try:
        a,b,record=training.verify('diffusability_random24_VGG_frozen',model,original,lambda net,s:method.training(net,images,None,STREAM).loss,steps)
    finally:training.AUDIT_FACTORY=saved_factory
    # Independent source utility comparisons, including their input gradients.
    from umdiff.regularizers import DCTScaleEquivariance
    utilities=native_dct_utilities();x=torch.randn(2,4,8,8,requires_grad=True);rgb=torch.randn(2,4,16,16)
    for zigzag in (False,True):
        training.seed(97);ratio=np.random.rand()*.5;z=utilities['cut_dct2d_high_freqs'](x,block_size=4,zigzag=zigzag,cut_ratio=ratio);target=utilities['cut_dct2d_high_freqs'](rgb,block_size=8,zigzag=zigzag,cut_ratio=ratio);expected=(F.interpolate(z,scale_factor=2,mode='nearest')-target).square().flatten(1).mean(1)
        training.seed(97);actual=DCTScaleEquivariance(block_size=4,compression=2,zigzag=zigzag)(lambda v:F.interpolate(v,scale_factor=2,mode='nearest'),x,rgb);CHECK('diffusability/DCT_cut'+str(zigzag),actual,expected,atol=0,rtol=0)
        CHECK('diffusability/DCT_cut_gradient'+str(zigzag),torch.autograd.grad(actual.sum(),x,retain_graph=True)[0],torch.autograd.grad(expected.sum(),x,retain_graph=True)[0],atol=0,rtol=0)
    record['perceptual_scope']='Actual author VGG19 with synthetic weights. Three VGG scales and full SD-style AutoencoderKL/256-grid for --full-codec; otherwise reduced64-grid. No pretrained or paper quality claim.'
    record['full_codec_architecture']=FULL_CODEC
    return record

CASES={'dctdiff':dctdiff,'mdm':mdm,'mdm_vdm':lambda steps:mdm(steps,True),'ssd':ssd,'ppflow':ppflow,'dimr':dimr,'dfm':dfm,'lapflow':lapflow,'lapflow3':lapflow3,'relay':relay,'rdd':rdd,'diffusability':diffusability}

def main():
    global OUT,FULL_CODEC
    parser=argparse.ArgumentParser();parser.add_argument('method',choices=CASES);parser.add_argument('--steps',type=int,default=128);parser.add_argument('--directory',type=Path);parser.add_argument('--full-codec',action='store_true');parser.add_argument('--threads',type=int,default=2);args=parser.parse_args()
    FULL_CODEC=args.full_codec
    if args.directory:OUT=args.directory.resolve();conformance.ARTIFACT_ROOT=OUT/'artifacts'
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads);training.seed(20261009);audits=[]
    def factory(name,pairs):
        audit=ActivationAudit(name,pairs,OUT/'activation',excluded_modules=('',) if args.method=='diffusability' else ());audits.append(audit);return audit
    training.AUDIT_FACTORY=factory;training.CHECKPOINT_ROOT=ROOT/'runs'/OUT.relative_to(ROOT/'evidence')/'native_pairs'
    started=time.time();details=None;failure=None
    try:details=CASES[args.method](args.steps)
    except Exception:failure=traceback.format_exc();print(failure,flush=True)
    activation=[a.close() for a in audits]
    mismatched=[a['name'] for a in activation if a['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in a['totals'].values())]
    if mismatched and failure is None:failure='Module activation/gradient audit failed: '+', '.join(mismatched)
    nonbyte=[c['name'] for c in conformance.CHECKS if c.get('bitwise') is False]
    if nonbyte and failure is None:failure='Strict component/loss/state byte comparison failed: '+', '.join(nonbyte[:10])
    import hashlib
    report={'method':args.method,'status':'failed' if failure else 'passed','steps':args.steps,'device':'cpu','synthetic':True,'gpu_used':False,'seconds':time.time()-started,'details':details,'failure':failure,'checks':conformance.CHECKS,'activation':activation,'source_executions':native.AUDIT,'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'implementation_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'umdiff').glob('*.py'))},'scope':('Full SD-style AutoencoderKL, author VGG19 at three scales, synthetic256-grid; frozen encoder. ' if FULL_CODEC else 'Reduced author networks; ')+ 'Specified enabled branches; shared native backbones. No pretrained/CLI/paper-quality result.'}
    (OUT/(args.method+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps({k:report[k] for k in ('method','status','seconds','failure')}),flush=True)
    return bool(failure)
if __name__=='__main__':raise SystemExit(main())
