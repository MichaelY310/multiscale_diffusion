#!/usr/bin/env python3
"""Export released architectures/recipes separately from numerical fixtures."""
import sys,json,hashlib,copy
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import numpy as np
from umdiff import native
OUT=ROOT/'configs/paper_recipes';OUT.mkdir(parents=True,exist_ok=True)
INDEX=[]
def obj(x):
    if hasattr(x,'to_dict'):return obj(x.to_dict())
    if isinstance(x,dict):return {k:obj(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [obj(v) for v in x]
    if hasattr(x,'__dict__'):return obj(vars(x))
    return x
def base(name,shape,model,optimizer=None):
    return {'name':name,'image_shape':shape,'model':model,'training':{'optimizer':optimizer or {'lr':1e-4,'betas':[.9,.999],'weight_decay':0.,'ema_decay':.9999},'data':{'shuffle':True}}}
def save(name,c,sources,notes=(),batch=None,steps=None):
    if c.get('task')!='rmd':c['training']['optimizer']['randomness']='native_global'
    if name.startswith(('dctdiff_','dimr_','ppflow_','dfm_','pixelflow_','lapflow_','rdd_')):
        c['training']['data']['drop_last']=True
    pins=[{'file':s,'sha256':hashlib.sha256((ROOT/'sources'/s).read_bytes()).hexdigest()} for s in sources]
    c['evidence']={'claim':'released_recipe_template','full_recipe_quality_verified':False,'synthetic_training_allowed':False,'source_pins':pins,'notes':list(notes),'author_global_batch':batch,'author_steps':steps}
    if name.startswith(('lapflow_','diffusability_')):c['evidence']['claim']='component_recipe_template_best_run_unidentified'
    if name.startswith('rdd_'):c['evidence']['claim']='local_rdd_recipe_template'
    if name.startswith('rdd_'):
        c['evidence']['source_pins']=[{'root':'workspace','file':file,'sha256':hashlib.sha256((ROOT.parent/file.removeprefix('multiscale_diffusion/')).read_bytes()).hexdigest()} for file in ('multiscale_diffusion/2d/train_RDD_imagenet256_unified.py','multiscale_diffusion/2d/sample_fid_unified.py','multiscale_diffusion/DiT/models_multires.py')]
    (OUT/(name+'.json')).write_text(json.dumps(c,indent=2)+'\n');INDEX.append({'name':name,'path':str((OUT/(name+'.json')).relative_to(ROOT)),**c['evidence']})
def labels(key='y',dropout=0.,rng='torch'):
    return {key:{'kind':'labels','classes':1000,**({'dropout':dropout,'dropout_rng':rng,'null_label':1000} if dropout else {})}}
def latent(c):
    c['training']['data'].update(image_size=256,crop='adm_center',horizontal_flip=True,flip_rng='torch',normalization='torchvision')
    c['training']['input_transform']={'name':'local_vae','args':{'model':{'name':'local_autoencoder','args':{'path':'@vae@'}},'scale':.18215}}
def adapter(model,**kw):return {'name':'adapter','args':{'model':model,**kw}}

# Every released DCT U-ViT recipe; retain actual EBFR statistics and sampler.
for file in sorted((ROOT/'sources/dctdiff/configs').glob('*.py')):
    cfg=native.load('dctdiff/configs/'+file.name).get_config();d=cfg.dataset
    opt=obj(cfg.optimizer);opt['optimizer_name']=opt.pop('name');opt.update(ema_decay=.9999,ema_style='scaled_add',lr_schedule={'name':'warmup','warmup_steps':cfg.lr_scheduler.warmup_steps,'offset':0})
    net=obj(cfg.nnet);net.pop('name');c=base('dctdiff',[3,d.resolution,d.resolution],{'name':'native_dctdiff','args':net},opt)
    statistics={k:obj(d[k]) for k in ('Y_std','Cb_std','Cr_std')};statistics.update(block=d.block_sz,frequencies=d.low_freqs)
    c.update(representation={'name':'dct_ycbcr','args':{'resolution':d.resolution,'block':d.block_sz,'frequencies':d.low_freqs,'bound':obj(d.Y_bound)}},path={'name':'vp','args':{'snr_scale':d.SNR_scale}},target={'name':'joint','args':{'quantity':'epsilon'}},clock={'name':'linear','args':{'network_scale':999.}},metric={'name':'mse','args':{'dct_statistics':statistics}},sampler={'name':'dctdiff_dpm' if cfg.sample.algorithm=='dpm_solver' else 'dctdiff_euler_maruyama','args':{'steps':cfg.sample.sample_steps,'snr_scale':d.SNR_scale}})
    if cfg.sample.algorithm=='euler_maruyama_sde':c['sampler']['args']['stochastic']=True
    c['training']['optimizer']['precision']='float16'
    c['training']['time_sampler']={'name':'uniform'};c['training']['data'].update(crop='pre_resized',horizontal_flip=True,flip_rng='torch')
    if net['num_classes']>0:c['training']['conditioning']=labels()
    save('dctdiff_'+file.stem,c,['dctdiff/configs/'+file.name,'dctdiff/train.py'],['Original split/augmentation and dataset files must be supplied locally; CIFAR requires an exported local image folder. No claim to reproduce a published FID.'],cfg.train.batch_size,cfg.train.n_steps)

betas=(np.linspace(.00085**.5,.012**.5,1000)**2).tolist()
for file in sorted((ROOT/'sources/dimr/configs').glob('*.py')):
    cfg=native.load('dimr/configs/'+file.name).get_config();opt=obj(cfg.optimizer);opt['optimizer_name']=opt.pop('name');opt.update(ema_decay=.9999,ema_style='scaled_add',lr_schedule={'name':'warmup','warmup_steps':cfg.lr_scheduler.warmup_steps,'offset':0})
    opt['precision']='float16'
    c=base('dimr',obj(cfg.z_shape),adapter({'name':'native_dimr','args':{'config':obj(cfg.nnet.model_args)}},time_keyword='log_snr'),opt)
    c.update(batch={'name':'dimr','args':{'betas':betas,'loss_weights':obj(cfg.loss_coeffs)}},sampler={'name':'dimr_dpm','args':{'betas':betas,'steps':cfg.sample.sample_steps,'guidance_scale':cfg.sample.scale,'guidance_power':-1}})
    c['training'].update(time_sampler={'name':'discrete','args':{'steps':1000,'offset':1,'rng':'numpy'}},conditioning=labels(dropout=cfg.dataset.p_uncond,rng='python'),input_transform={'name':'vae_moments','args':{'scale':.18215}},data={'kind':'moments_npy','shuffle':True})
    save('dimr_'+file.stem,c,['dimr/configs/'+file.name,'dimr/train.py','dimr/eval.py'],['Input: native numbered .npy mean/logvar+class records. Latent sampling re-runs every epoch. RGB decoding requires the original local VAE.'],cfg.train.batch_size,cfg.train.n_steps)

import yaml
p=yaml.safe_load((ROOT/'sources/pixelflow/configs/pixelflow_xl_c2i.yaml').read_text());c=base('pixelflow',[3,256,256],adapter({'name':'native_pixelflow','args':p['model']['params']},time_keyword='timestep'),{'lr':float(p['train']['lr']),'weight_decay':p['train']['weight_decay'],'ema_decay':.9999,'grad_clip':1.,'precision':'bfloat16'})
c.update(batch={'name':'pixelflow','args':{'stages':4,'patch':p['model']['params']['patch_size'],'attention_head_dim':p['model']['params']['attention_head_dim']}},sampler={'name':'pixelflow','args':{'stages':4,'steps':[30]*4,'guidance_scale':2.4}})
c['training'].update(time_sampler={'name':'discrete','args':{'steps':1000}},conditioning={'class_labels':{'kind':'labels','classes':1000,'destination':'context'}},data={'crop':'pixelflow_random','horizontal_flip':True,'flip_rng':'torch','normalization':'torchvision','expand_ratio':p['data']['expand_ratio'],'shuffle':True})
save('pixelflow_xl_c2i',c,['pixelflow/configs/pixelflow_xl_c2i.yaml','pixelflow/train.py','pixelflow/sample_ddp.py','pixelflow/pixelflow/data_in1k.py'],['Source recipe: 10 epochs, batch4 per rank; sample_ddp.py defaults30 steps per stage and CFG2.4. These public defaults are not certified optimal FID settings. GPU FlashAttention/bf16 is not covered by CPU SDPA parity.'],steps='10 epochs')

for levels in (2,3):
    c=base('ppflow',[4,32,32],{'name':'native_ppflow','args':{'input_size':32,'patch_size':2,'patch_level':levels,'in_channels':4,'hidden_size':1152,'depth':28,'num_heads':16,'class_dropout_prob':.1,'learn_sigma':True}})
    c.update(batch={'name':'ppflow','args':{'levels':levels}},sampler={'name':'ppflow_native','args':{'points':250,'sampling_method':'heun2','cfg_scales':[1.,3.] if levels==2 else [1.,3.5,3.75]}});latent(c)
    c['training'].update(time_sampler={'name':'none'},conditioning=labels(),initialization={'mode':'ppflow_sit','path':'@sit@'})
    save('ppflow_xl_'+str(levels),c,['ppflow/train.py','ppflow/models.py','ppflow/sample_ddp.py','ppflow/train_utils.py'],['Released 2/3-stage architecture and sample_ddp ODE policy (250 saved points, Heun2). CFG[1,3] is the2-stage eval default; [1,3.5,3.75] is the source3-stage example, not a certified optimal run. Supply author SiT initialization. Unreleased 4/5-stage and FLUX variants are deferred.'],256,'1400 epochs')

c=base('dfm',[4,32,32],adapter({'name':'native_dfm','args':{'input_size':32,'in_channels':4,'hidden_size':1152,'decoder_hidden_size':1152,'encoder_depth':8,'depth':28,'num_heads':16,'class_dropout_prob':.1,'enable_repa':False,'qk_norm':False}},output_index=0),{'lr':1e-4,'weight_decay':0.,'ema_decay':.9999,'grad_clip':1.,'precision':'float16'})
c.update(batch={'name':'dfm','args':{'native_debug_rng':True}},representation={'name':'laplacian','args':{'levels':2,'up':'bilinear','order':'coarse_first'}},clock={'name':'dfm','args':{'steps':[40,10],'thresholds':[.1]}},guidance={'name':'cfg_window','args':{'scale':1.5,'low':0.,'high':1.}});latent(c);c['training'].update(time_sampler={'name':'dfm'},conditioning=labels())
save('dfm_sit_xl',c,['dfm/train.py','dfm/models/sit_dfm.py','dfm/README.md'],['Released SiT reimplementation explicitly differs from the paper architecture. REPA objective/projector is available, but a real local teacher feature pipeline must be provided for REPA experiments.'],256,400000)
repa=copy.deepcopy(c);repa['model']=repa['model']['args']['model'];repa['model']['args'].update(enable_repa=True,z_dims=[768],projector_dim=2048);repa['batch']['args'].update(repa_weight=.3);repa['training'].pop('input_transform');repa['training']['conditioning']['teacher_features']={'kind':'field','field':'teacher_features','wrap_list':True,'destination':'context','phase':'train'}
save('dfm_sit_xl_repa_localfeatures',repa,['dfm/train.py','dfm/models/sit_dfm.py','dfm/loss.py'],['Local .pt contains latents, labels, conditions.teacher_features from the desired DINO teacher. This avoids constructing an unverified or downloaded teacher. VAE/teacher feature preprocessing must match the source dataset pipeline.'],256,400000)

dfm_b=copy.deepcopy(c);dfm_b['model']['args']['model']['args'].update(hidden_size=768,decoder_hidden_size=768,depth=12,num_heads=12)
dfm_b['batch']['args'].update(weighting='lognormal')
dfm_b['training']['time_sampler']={'name':'dfm','args':{'weighting':'lognormal'}}
dfm_b['training']['optimizer']['precision']='bfloat16'
dfm_b['clock']['args'].update(steps=[30,10])
save('dfm_sit_b_released_yaml',dfm_b,['dfm/experiments/train/dfm_sit_b_256.yaml','dfm/experiments/generation/dfm_sit_b_256.yaml','dfm/train.py','dfm/loss.py'],['Exact released B training/generation YAML hyperparameters include lognormal active-stage time sampling and BF16. The native SiT release explicitly differs from the paper architecture; this is not a paper-best architecture claim.'],256,400000)

c=base('lapflow',[4,32,32],adapter({'name':'native_lapflow','args':{'input_size':[32,16],'patch_size':2,'in_channels':4,'hidden_size':1152,'depth':28,'num_heads':16,'class_dropout_prob':.1,'use_mot':True,'use_mask':True}},time_first=True),{'lr':5e-4,'weight_decay':0.,'ema_decay':.9,'grad_clip':1.})
c.update(batch={'name':'lapflow','args':{'levels':2,'activations':[.9,0.]}},sampler={'name':'lapflow','args':{'levels':2,'activations':[.9,0.],'step_size':.01,'guidance_scale':2.}});latent(c);c['training'].update(time_sampler={'name':'lapflow','args':{'activations':[.9,0.]}},conditioning=labels())
save('lapflow_mot_xl_2',c,['lapflow/feb12_train_ours.py','lapflow/models/MultiScaleDiT.py'],['CLI default architecture/optimizer profile, not an identified best paper run. Source epoch cosine schedule requires dataset-dependent steps_per_epoch. Broken released square sampling remains explicitly deferred.'])

lap_best=copy.deepcopy(c);lap_best['training']['optimizer'].update(lr=1e-4,ema_decay=.9999)
lap_best['batch']['args']['activations']=[.5,0.]
lap_best['sampler']['args']['activations']=[.5,0.]
lap_best['training']['time_sampler']['args']['activations']=[.5,0.]
lap_best['training']['input_transform']['args']['model']['args']['path']='@eqvae@'
save('lapflow_imagenet_xl_2_paper',lap_best,['lapflow/feb12_train_ours.py','lapflow/feb12_test_ours.py','lapflow/models/MultiScaleDiT.py'],['Paper AppendixG ImageNet256 XL/2: LR1e-4, constant schedule, EMA.9999, two scales, T=.5, EQVAE, globalbatch256,120epochs. Repo CLI defaults(T=.9/LR5e-4/EMA.9) are NOT this paper profile.','Paper source: https://arxiv.org/html/2602.19461#A7 . EQVAE must be supplied locally; VAE substitution changes the experiment. CFG2 remains an explicit evaluation choice rather than a certified best-FID scale.'],256,'120 epochs')

# SSD: released train_ssd.sh overrides the YAML predict_xstart=False default.
from umdiff.native_models import activate
activate('ssd',['guided_diffusion'])
from guided_diffusion.ssd_utils import get_resolution_scale_space
grid=[get_resolution_scale_space([8,16,32,64],t,1000,'ConvexDecay_0.5') for t in range(1000)];beta=np.linspace(.0001,.02,1000).tolist();mapping={8:8,16:16,32:32,64:64}
c=base('ssd',[3,64,64],{'name':'native_ssd','args':{'image_size':64,'in_channels':3,'model_channels':128,'out_channels':3,'num_res_blocks':2,'attention_resolutions':[4,8],'channel_mult':[1,2,3,4],'num_heads':4,'use_scale_shift_norm':True,'unet_res_layer_map':mapping}})
c.update(batch={'name':'ssd','args':{'resolutions':grid,'betas':beta,'unet_res_layer_map':mapping,'loss_weight':'min_snr_5'}},prediction_adapter={'name':'ssd_output'},sampler={'name':'ssd','args':{'resolutions':grid,'betas':beta,'unet_res_layer_map':mapping}})
c['training'].update(time_sampler={'name':'ssd','args':{'resolutions':grid}},data={'crop':'adm_center','horizontal_flip':True,'flip_rng':'python','shuffle':True})
save('ssd_celeba64',c,['ssd/bash_scripts/train_ssd.sh','ssd/configs/model.yaml','ssd/configs/training.yaml','ssd/configs/diffusion.yaml','ssd/configs/ssd.yaml'],['Source script uses CelebA train split and batch32 per rank on2 ranks (global64); supply that exact split. Source script does not specify a total training budget.'],64)

# MDM's text encoder is an offline data prerequisite, not a diffusion target.
from umdiff import build
from umdiff.native_models import import_mdm
import_mdm()
from ml_mdm import samplers
gamma=samplers.Sampler(samplers.SamplerConfig(num_diffusion_steps=1000,schedule_type=samplers.ScheduleType.DEEPFLOYD)).gammas.cpu().tolist()
cfg=yaml.safe_load((ROOT/'sources/mdm/ml-mdm-matryoshka/configs/models/cc12m_256x256.yaml').read_text());network=copy.deepcopy(cfg['unet_config'])
network['conditioning_feature_dim']=2048;network['inner_config']['conditioning_feature_dim']=2048
if network.get('initialize_inner_with_pretrained')=='None':network['initialize_inner_with_pretrained']=None
c=base('mdm',[3,256,256],{'name':'native_mdm','args':{'input_channels':3,'output_channels':3,'config':network}},{'lr':cfg['lr'],'optimizer_name':'adam','weight_decay':0.,'ema_decay':.9999,'grad_clip':2.,'lr_schedule':{'name':'warmup','warmup_steps':10000,'offset':0,'minimum_step':1}})
c['training']['optimizer']['accumulation_normalization']='sum'
c.update(batch={'name':'mdm','args':{'gammas':gamma,'scales':[4,1],'mixed_ratio':'2:1'}},prediction_adapter={'name':'mdm_loss_coordinates'},sampler={'name':'mdm','args':{'gammas':gamma,'scales':[4,1],'threshold':'clip','resample_steps':True,'inference_steps':250,'eta':0.,'guidance_scale':7.5,'clip_output':True}})
c['training'].update(time_sampler={'name':'discrete','args':{'steps':1000,'offset':1}},conditioning={'conditioning':{'kind':'field','field':'lm_outputs'},'cond_mask':{'kind':'field','field':'lm_mask'},'micros':{'kind':'empty_dict'}})
c['training']['conditioning']['unconditional']={'kind':'mapping','phase':'sample','fields':{'conditioning':{'kind':'field','field':'negative_lm_outputs'},'cond_mask':{'kind':'field','field':'negative_lm_mask'},'micros':{'kind':'empty_dict'}}}
save('mdm_cc12m256',c,['mdm/ml-mdm-matryoshka/configs/models/cc12m_256x256.yaml','mdm/ml-mdm-matryoshka/ml_mdm/clis/train_parallel.py','mdm/ml-mdm-matryoshka/ml_mdm/clis/generate_sample.py'],['Local input .pt: images, conditions.lm_outputs (Flan-T5-XL), conditions.lm_mask. Sampling requires matching negative_lm_outputs/negative_lm_mask. Released demo defaults:250 resampled steps, eta0, CFG7.5, clip; not an identified best FID setting.','Source starts from a pretrained vision model; add an explicit local training.initialization when reproducing that fine-tuning run. CC12M streaming and original training curriculum are not certified.'],steps=1000000)

cfg1024=yaml.safe_load((ROOT/'sources/mdm/ml-mdm-matryoshka/configs/models/cc12m_1024x1024.yaml').read_text())
network1024=copy.deepcopy(cfg1024['unet_config']);network1024['conditioning_feature_dim']=2048
network1024['initialize_inner_with_pretrained']='@mdm_inner256@'
c1024=copy.deepcopy(c);c1024['image_shape']=[3,1024,1024]
c1024['model']['args']['config']=network1024
c1024['batch']['args']={'gammas':gamma,'scales':[16,4,1],'power':2.,'weights':[16.,4.,1.]}
c1024['sampler']['args'].update(scales=[16,4,1],power=2.)
c1024['training']['optimizer']['precision']='float16'
c1024['training']['optimizer']['accumulation_normalization']='mean'
save('mdm_cc12m1024',c1024,['mdm/ml-mdm-matryoshka/configs/models/cc12m_1024x1024.yaml','mdm/ml-mdm-matryoshka/ml_mdm/models/nested_unet.py','mdm/ml-mdm-matryoshka/ml_mdm/diffusion.py'],['Released1024 three-level config includes loss weights16:4:1, SNR shift power2 and local inner256 initialization. Supply that exact inner checkpoint and T5 features; random initialization is not a substitute for this curriculum.','Streaming reader/curriculum/precision and pretrained quality are not certified by meta construction or random-weight short runs.'],steps=1000000)

# RDD multires training uses independently assembled objectives in one program.
from umdiff.rdd_native import diffusion_backend
backend=diffusion_backend();resolutions=[32,16,8];limits=[0,300,600,1000];stages=[]
for i,res in enumerate(resolutions):
    d=backend.create_diffusion_with_drift('',diffusion_steps=1000,c=i if i<2 else None,T_c=limits[i],T_c_plus_1=limits[i+1],sample_scale=2)
    path={'betas':d.betas.tolist(),'timestep_map':d.timestep_map,'drift_start':limits[i] if i<2 else None,'drift_end':limits[i+1] if i<2 else None}
    stages.append({'input':{'name':'average_pool','args':{'factor':2**i,'detach':i>0}},'time_sampler':{'name':'discrete','args':{'steps':d.num_timesteps}},'method':{'batch':{'name':'rdd_ddpm','args':{'path':path,'learned_variance':True}}}})
c=base('rdd',[4,32,32],{'name':'native_rdd_unified','args':{'patch_size':2,'in_channels':4,'hidden_size':1152,'depth':28,'num_heads':16,'class_dropout_prob':.1,'learn_sigma':True,'base_size':16}})
c.update(training_stages=stages,training_stage_reduction='sum_stage_means',sampler={'name':'rdd_native_stages','args':{'resolutions':resolutions,'stage_lengths':[300,300,400],'steps':[100,75,75],'solver':'dpm2','boundary':'fresh','guidance_scale':1.5}});latent(c);c['training'].update(time_sampler={'name':'none'},conditioning=labels())
save('rdd_unified_xl_300_300_400',c,[],['Local migrated train_RDD_imagenet256_unified.py architecture and unweighted multi-stage objective. Native solver backend explicitly reused. REPA/min-SNR/loss-aware sampling options of the local extended trainer are not all certified here.'])

c=base('diffusability',[3,256,256],{'name':'local_autoencoder','args':{'path':'@vae@'}},{'lr':1e-4,'ema_decay':0.})
c.update(task='codec',sampling_shape=[4,32,32],codec={'reconstruction_style':'native','mse_weight':0.,'kl_weight':0.,'perceptual':{'weight':1.,'model':{'name':'native_perceptual_pyramid','args':{'checkpoint':'@vgg@','scales':[1.,.5,.25],'loss_weights':[1,1,1,1,1]}}},'regularizer_ignore_perceptual':False,'regularizers':[{'name':'random_scale_equivariance','args':{'factors':[2,4],'weight':1.}}]});c['training']['time_sampler']={'name':'none'}
save('diffusability_se24',c,['diffusability/configs/loss/autoencoder.yaml','diffusability/configs/loss/base.yaml'],['SE 2x/4x extension of the released VGG-pyramid codec objective. SE weight, optimizer/budget, dataset and encoder freezing vary across paper experiments; this is a component-complete template, not an identified best run.','Temporal/tokenizer/discriminator/distillation-AE and downstream generator retraining remain separate pipelines.'])

for stage,size in [('low',64),('high',256)]:
    net={'img_resolution':size,'img_channels':3,'model_type':'DhariwalUNet','model_channels':192 if stage=='low' else 256,'channel_mult':[1,2,3,4] if stage=='low' else [1,1,2,2,4,4],'num_blocks':3 if stage=='low' else 2,'dropout':.1,'label_dim':1000,'augment_dim':9,'use_fp16':True}
    # README uses 8 ranks, global batch4096, microbatch32/8. Source DDP averages
    # per-rank summed gradients; Trainer averages a logical local batch512.
    c=base('relay_'+stage,[3,size,size],{'name':'native_relay','args':net}, {'lr':1e-4,'optimizer_name':'adam','weight_decay':0.,'image_count_multiplier':8,'accumulation_steps':16 if stage=='low' else 64,'ema_style':'source_lerp','ema_schedule':{'name':'image_halflife','half_life_images':50000000 if stage=='low' else 500000,'rampup_ratio':.05},'lr_schedule':{'name':'image_warmup','warmup_images':10000000},'loss_multiplier':3*size*size*512*(25 if stage=='low' else 1)/(32 if stage=='low' else 256)})
    c.update(batch={'name':'edm'} if stage=='low' else {'name':'relay','args':{'block_scale':.15,'prob_length':.93,'blur_max':3.}},sampler={'name':'relay','args':{'steps':200,'rho':100.,'cfg':3.5,'block_scale':.15}} if stage=='high' else {'name':'edm','args':{'num_steps':256,'S_churn':40,'S_min':.05,'S_max':50,'S_noise':1.003}})
    c['training']['optimizer']['gradient_repair']={'nan':0.,'posinf':1e5,'neginf':-1e5}
    c['batch'].setdefault('args',{})['augmentation']={'p':.12,'xflip':1e8,'yflip':1,'scale':1,'rotate_frac':1,'aniso':1,'translate_frac':1}
    c['batch']['args']['native_loss']=True
    c['training']['optimizer'].update(loss_reduction='sum',loss_multiplier=25/32 if stage=='low' else 1/256)
    c['training'].update(time_sampler={'name':'none'} if stage=='low' else {'name':'uniform'},conditioning={'class_labels':{'kind':'labels','classes':1000,'one_hot':True}})
    save('relay_imagenet_'+stage,c,['relay/README.md','relay/train.py','relay/generate_imagenet.py','relay/training/networks.py','relay/training/loss.py','relay/training/training_loop.py'],['README ImageNet performance-reproduction architecture/training flags and generate_imagenet.py sampling defaults. Supply original preprocessed data and both pretrained networks for best-result inference.','Global batch4096 on8 ranks is part of the source recipe. This single-process Trainer still requires distributed image-count/global-batch adaptation before that full recipe is certified; loss_multiplier encodes logical per-rank batch512. Recompute for any different world size/batch. Source nan_to_num gradient repair is explicitly configured.'],4096)

c={'name':'rmd','task':'rmd','image_shape':[16,21,60,104],'model':{'name':'local_wan_transformer','args':{'path':'@wan@'}},'training':{'optimizer':{'lr':5e-6,'fake_lr':2e-5,'betas':[0.,.95],'weight_decay':1e-4,'eps':1e-8,'grad_clip':1.},'precision':'bfloat16','accumulation_steps':2},'distillation':{'k_step':6,'eta':.5,'flow_shift':3.,'flow_shift_trans':1,'low_rs_step':3,'warmup_step':15,'split_timestep':True,'t_offset_factor':.25,'target_size':[90,160],'cfg':5.,'score_weighting_mode':'inverse_sigma'}}
save('rmd_wan13b',c,['rmd/configs/train_wan_1_3b.yaml'],['Released Wan2.1 1.3B recipe; supply local prompt embeddings, negative embeddings and Wan transformer. Exact latent dimensions must match the selected source CLI height/width/frame count. BF16/GPU quality, optional GAN and alternative backbones are not certified.','Repository attribution remains unconfirmed by an author-linked paper URL. Numerical claims concern the pinned release.'],8,300)

(OUT/'INDEX.json').write_text(json.dumps(INDEX,indent=2)+'\n')
print(json.dumps({'recipes':len(INDEX),'directory':str(OUT)}))
