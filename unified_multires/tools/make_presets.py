"""Generate reviewable small native-algorithm and controlled configurations."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
import torch

def component(name,**args):return {'name':name,**({'args':args} if args else {})}
def adapter(model,**kwargs):return component('adapter',model=model,**kwargs)
def main():
    common={'optimizer':{'lr':1e-4,'betas':[.9,.99],'weight_decay':.01,'ema_decay':.99,'grad_clip':1.},'time_sampler':component('uniform')}
    configs={}
    def add(name,shape,model,**method):
        config={'name':name,'image_shape':shape,'model':model,'training':json.loads(json.dumps(common)),**method}
        config['evidence']={'claim':'native_algorithm_small_architecture_fixture','not_a_paper_quality_checkpoint':True,'no_downloads':True}
        configs[name]=config;return config
    def labels(config,key='y',context=False):config['training']['conditioning']={key:{'kind':'labels','classes':1000,**({'destination':'context'} if context else {})}}
    tiny=component('tiny_conv',channels=3,width=32)
    flow=add('ordinary_flow',[3,32,32],tiny,path=component('flow'),target=component('joint',quantity='velocity'))
    flow['evidence']['claim']='controlled_backbone_reference'
    for learned,decoupled in ((False,False),(True,False),(False,True),(True,True)):
        path={'steps':16,'beta_start':.001,'beta_end':.02,'drift_start':0,'drift_end':16}
        config=add('rdd_'+('decoupled' if decoupled else 'joint')+('_learnedvar' if learned else ''),[4,16,16],component('native_rdd',input_size=16,patch_size=2,in_channels=4,hidden_size=32,depth=2,num_heads=4,class_dropout_prob=0.,learn_sigma=learned,d_prediction=decoupled),batch=component('rdd_ddpm',path=path,learned_variance=learned,decoupled=decoupled),sampler=component('rdd_ddpm',path=path,learned_variance=learned,decoupled=decoupled))
        config['training']['time_sampler']=component('discrete',steps=16);labels(config)
    path={'steps':16,'beta_start':.001,'beta_end':.02}
    config=add('ordinary_ddpm',[3,32,32],tiny,batch=component('rdd_ddpm',path=path),sampler=component('rdd_ddpm',path=path));config['training']['time_sampler']=component('discrete',steps=16);config['evidence']['claim']='controlled_backbone_reference'
    add('relay',[3,16,16],component('native_relay',img_resolution=16,img_channels=3,model_channels=64,channel_mult=[1,2],num_blocks=1,dropout=0.,attn_resolutions=[8]),batch=component('relay',patch=4,block_scale=.15),target=component('joint',quantity='clean'),sampler=component('relay',steps=12,sigma_max=2.,truncation_sigma=1.))
    add('dctdiff',[3,32,32],component('native_dctdiff',embed_dim=32,depth=2,num_heads=4,tokens=16,low_freqs=4,num_classes=-1),representation=component('dct_ycbcr',resolution=32,block=4,frequencies=4,bound=256.),path=component('vp',snr_scale=4.),target=component('joint',quantity='epsilon'),clock=component('linear',network_scale=999.),sampler=component('dctdiff_dpm',steps=12,snr_scale=4.))
    for levels in (2,3):
        activations=[.4,0.] if levels==2 else [.6,.3,0.]
        config=add(f'lapflow_{levels}',[4,16,16],adapter(component('native_lapflow',input_size=[16//2**i for i in range(levels)],patch_size=2,in_channels=4,hidden_size=32,depth=2,num_heads=4,class_dropout_prob=0.,use_mot=True,use_mask=True),time_first=True),batch=component('lapflow',levels=levels),sampler=component('lapflow',levels=levels,step_size=.1));config['training']['time_sampler']=component('lapflow',activations=activations);labels(config)
    config=add('dfm',[4,16,16],adapter(component('native_dfm',input_size=16,in_channels=4,hidden_size=32,decoder_hidden_size=32,encoder_depth=1,depth=2,num_heads=4,class_dropout_prob=0.,enable_repa=False,qk_norm=False),output_index=0),batch=component('dfm'),representation=component('laplacian',levels=2,up='bilinear',order='coarse_first'),clock=component('dfm'));labels(config)
    repa=json.loads(json.dumps(config));repa['name']='dfm_repa'
    repa['model']=component('native_dfm',input_size=16,in_channels=4,hidden_size=32,decoder_hidden_size=32,encoder_depth=1,depth=2,num_heads=4,class_dropout_prob=0.,enable_repa=True,z_dims=[8],projector_dim=32,qk_norm=False)
    repa['batch'].setdefault('args',{})['repa_weight']=.3
    repa['training']['conditioning']['teacher_features']={'kind':'zeros','shape':[64,8],'destination':'context','phase':'train'}
    repa['evidence']['teacher_features']='Synthetic zero teacher fixture; set kind=field and field=teacher_features with a local tensor dataset for actual REPA training.'
    configs['dfm_repa']=repa
    # DFMBatch receives original images and constructs its own representation;
    # Method representation is used to decode product-state samples only.
    for levels in (2,3):
        config=add(f'ppflow_{levels}',[4,16,16],component('native_ppflow',input_size=16,patch_size=2,patch_level=levels,in_channels=4,hidden_size=32,depth=2,num_heads=4,class_dropout_prob=0.,learn_sigma=True),batch=component('ppflow',levels=levels),sampler=component('ppflow',steps=12));config['training']['time_sampler']=component('none');labels(config)
    config=add('pixelflow',[3,32,32],adapter(component('native_pixelflow',in_channels=3,out_channels=3,num_attention_heads=2,attention_head_dim=16,depth=2,patch_size=2,num_classes=1000),time_keyword='timestep'),batch=component('pixelflow',stages=3),sampler=component('pixelflow',stages=3,steps=4,guidance_scale=0.));config['training']['time_sampler']=component('discrete');labels(config,'class_labels',True)
    from umdiff.native_models import mdm
    mdm() # makes source imports available without downloading any assets
    from ml_mdm import samplers
    s=samplers.NestedSampler(samplers.SamplerConfig(num_diffusion_steps=16,schedule_type=samplers.ScheduleType.DEEPFLOYD,schedule_shifted=True))
    gammas=s.gammas.tolist()
    config=add('mdm',[4,16,16],component('native_mdm'),batch=component('mdm',gammas=gammas),prediction_adapter=component('mdm_loss_coordinates'),sampler=component('mdm',gammas=gammas));config['training']['time_sampler']=component('discrete',steps=16,offset=1);config['training']['conditioning']={'conditioning':{'kind':'zeros','shape':[5,8]},'cond_mask':{'kind':'ones','shape':[5]},'micros':{'kind':'empty_dict'}}
    config['evidence']['conditioning']='Zero text fixture; real training must switch conditioning/cond_mask to local tensor dataset field inputs.'
    def stage(block,dim,size,ratio):return dict(block_type=block,dim=dim,hidden_dim=2*dim,num_attention_heads=4,num_blocks=1,max_height=size,max_width=size,image_input_ratio=1,input_feature_ratio=ratio,final_kernel_size=3,kernel_size=3,dropout_prob=0)
    betas=(torch.linspace(.00085**.5,.012**.5,16,dtype=torch.float64)**2).tolist()
    config=add('dimr',[2,16,16],adapter(component('native_dimr',config=dict(channels=2,num_classes=1001,block_grad_to_lowres=False,norm_type='TDRMSN',stage_configs=[stage('TransformerBlock',32,8,2),stage('ConvNeXtBlock',16,16,1)])),time_keyword='log_snr'),batch=component('dimr',betas=betas),sampler=component('dimr_dpm',betas=betas,steps=12));config['training']['time_sampler']=component('discrete',steps=16,offset=1);labels(config)
    betas=torch.linspace(.001,.02,8,dtype=torch.float64).tolist();resolutions=[16]*4+[8]*4;mapping={8:8,16:16}
    config=add('ssd',[3,16,16],component('native_ssd',image_size=16,in_channels=3,model_channels=32,out_channels=3,num_res_blocks=1,attention_resolutions=[2],channel_mult=[1,2],unet_res_layer_map=mapping),batch=component('ssd',resolutions=resolutions,betas=betas,unet_res_layer_map=mapping),prediction_adapter=component('ssd_output'),sampler=component('ssd',resolutions=resolutions,betas=betas,unet_res_layer_map=mapping));config['training']['time_sampler']=component('discrete',steps=8,one_for_batch=True)
    # These are component transplants, not native author-model reproduction.
    config=add('rdd_flow_joint',[3,32,32],tiny,path=component('rdd_flow',start=0.,end=1.),target=component('joint',quantity='velocity'));config['evidence']['claim']='controlled_transplant'
    config=add('fdm_decoupled_clean',[3,32,32],component('tiny_conv',channels=3,width=32,heads=['signal','residual']),path=component('rdd_flow',start=0.,end=1.),target=component('decoupled_clean'));config['training']['time_sampler']=component('uniform',start=.01,end=.99);config['evidence']['claim']='paper_component_transplant_not_native_fdm'
    config=add('band_flow_epsilon',[3,32,32],tiny,path=component('flow'),target=component('joint',quantity='epsilon'),clock=component('linear',start=.001,end=.999));config['training']['time_sampler']=component('uniform',start=.001,end=.999);config['evidence']['claim']='controlled_target_swap'
    config=add('custom_plugin',[3,32,32],tiny,path={'name':'smooth_detail','module':'examples.custom_components'},target=component('joint',quantity='velocity'));config['evidence']['claim']='controlled_plugin_example'
    config=add('diffusability',[3,16,16],component('autoencoder_kl',block_out_channels=[32,32],down_block_types=['DownEncoderBlock2D']*2,up_block_types=['UpDecoderBlock2D']*2,latent_channels=4,norm_num_groups=32,layers_per_block=1),task='codec',sampling_shape=[4,8,8],codec={'mse_weight':1.,'kl_weight':.0001,'regularizers':[component('scale_equivariance',factor=2,mode='mean',weight=.3)]})
    config['evidence']['claim']='source_codec_pixel_mse_branch_no_perceptual_weights'
    config=add('rmd',[4,2,4,6],component('wan_transformer',patch_size=[1,2,2],num_attention_heads=2,attention_head_dim=16,in_channels=4,out_channels=4,text_dim=8,freq_dim=16,ffn_dim=64,num_layers=1),task='rmd',distillation={'k_step':6,'eta':.5,'flow_shift_trans':1,'low_rs_step':3,'warmup_step':3,'split_timestep':True,'t_offset_factor':.25,'target_size':[8,12]})
    config['training']={'optimizer':{'lr':5e-6,'fake_lr':2e-5,'betas':[0.,.95],'weight_decay':1e-4,'eps':1e-8,'grad_clip':1.}}
    config['evidence']['source_provenance']='Repository self-identifies official; paper does not link repository. Numerical claims refer to the pinned release, not independent author-identity confirmation.'
    for name,config in configs.items():
        directory=ROOT/'configs'/('controlled' if config['evidence']['claim'] in ('controlled_transplant','paper_component_transplant_not_native_fdm','controlled_target_swap','controlled_backbone_reference','controlled_plugin_example') else 'native_fixtures');directory.mkdir(parents=True,exist_ok=True)
        (directory/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n')
    print(len(configs),'training/sampling configs written')
    flux=component('flux_transformer',in_channels=16,num_layers=1,num_single_layers=1,num_attention_heads=2,attention_head_dim=16,joint_attention_dim=8,pooled_projection_dim=8,axes_dims_rope=[4,6,6])
    vae=component('autoencoder_kl',block_out_channels=[32,32],down_block_types=['DownEncoderBlock2D']*2,up_block_types=['UpDecoderBlock2D']*2,latent_channels=4,norm_num_groups=32,layers_per_block=1,scaling_factor=.3611,shift_factor=.1159)
    workflows={
      'speed':dict(name='speed',model=json.loads(json.dumps(flux)),vae=vae,sampler={'full_hw':[16,16],'scales':[.5,1.],'transitions':[.5],'steps':12,'transform':'dct'},initial_shape=[1,4,8,8],dtype='bfloat16'),
      'mrflow':dict(name='mrflow',model=flux,vae=vae,sr_model=component('native_rrdb',num_in_ch=3,num_out_ch=3,num_feat=8,num_block=2,num_grow_ch=4,scale=2),sampler={'low_size':32,'high_size':64,'low_steps':12,'refine_steps':1,'refine_sigma':.12}),
      'nami':dict(name='nami',model=component('native_nami',in_channels=64,num_layers=1,num_single_layers=3,num_attention_heads=1,attention_head_dim=16,joint_attention_dim=8,pooled_projection_dim=8,axes_dims_rope=[4,6,6],stage_singleblock_index=[1,2,3]),vae=component('autoencoder_kl',block_out_channels=[32]*4,down_block_types=['DownEncoderBlock2D']*4,up_block_types=['UpDecoderBlock2D']*4,latent_channels=16,norm_num_groups=32,layers_per_block=1,scaling_factor=.3611,shift_factor=.1159),sampler={'steps':[3,3,3],'guidance_scale':3.},initial_shape=[1,16,128,128])}
    workflows['speed']['model']['args']['guidance_embeds']=True
    directory=ROOT/'configs'/'inference_only';directory.mkdir(exist_ok=True)
    for name,config in workflows.items():
        config.update(synthetic_fixture=True,synthetic_embedding_shape=[1,3,8])
        (directory/f'{name}.json').write_text(json.dumps(config,indent=2)+'\n')
    print(len(workflows),'offline inference workflow configs written')
if __name__=='__main__':main()
