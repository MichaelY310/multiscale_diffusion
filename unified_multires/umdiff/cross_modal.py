"""Compile explicitly scoped mathematical recipes onto a selected backbone.

This does not rewrite the immutable author recipe/fixture configurations.
Model-domain tensors may be RGB or cached codec latents; no asset is fetched.
"""
import copy
import json
from pathlib import Path

STAGE_METHODS = ('rdd','fdm','pyramid_flow','pyramidal_wan_path','edify_laplacian','nonuniform_haar','pddpm','pixelflow','relay')
METHODS = (*STAGE_METHODS,'flow','ddpm','ssd','dfm','lapflow','mdm','dctdiff')
PRODUCT_METHODS = ('nonuniform_haar','dfm','lapflow','mdm')


def recipe(method, shape=(3,32,32), backbone='grid_dit', grids=None, width=96,
           depth=2, attention_heads=4, patch=(1,2,2), position='sincos',
           first_frame=False, separate_stages=False, decoupled=False,
           model_spec=None, process_args=None, steps=20, inference_policy=None, inference_args=None):
    if method not in METHODS:raise ValueError(f'Unknown method {method}; choose {METHODS}')
    shape=list(shape)
    if len(shape) not in (3,4) or min(shape)<1:raise ValueError('Shape must be C,H,W or C,T,H,W')
    if grids is None:
        grids=[[*shape[1:-2],shape[-2]//factor,shape[-1]//factor] for factor in ((4,1) if method=='relay' else (4,2,1))]
    grids=[list(g) for g in grids]
    if method not in STAGE_METHODS and first_frame:raise ValueError('This existing paper component preserves T; first-frame time pooling belongs to a stage process')
    if separate_stages and method not in STAGE_METHODS:raise ValueError('Separate stage models require the explicit stage program')
    if decoupled and method!='rdd':raise ValueError('--decoupled selects the RDD effective-epsilon/raw-drift objective')
    if process_args and method not in STAGE_METHODS:raise ValueError('Process arguments apply only to stage methods; edit the named batch/sampler components for other methods')
    channels=shape[0];ports=channels;heads=None
    product=method in PRODUCT_METHODS
    stages=len(grids) if method in STAGE_METHODS else 2 if product else 1
    if method=='fdm':
        heads=['signal' if (process_args or {}).get('prediction')=='clean' else 'epsilon','residual']
    if decoupled:heads=['main','drift']
    if method=='nonuniform_haar':
        from .grids import GridMap
        operators=[GridMap(b,a,first_frame) for a,b in zip(grids[:-1],grids[1:])]
        ports=[channels]+[channels*(2**sum(f==2 for f in op.factors)-1) for op in operators]
    if method=='dctdiff':
        if channels!=3 or shape[-2]!=shape[-1] or shape[-1]%8:raise ValueError('RGB DCTdiff requires 3 channels and square resolution divisible by 8')
        ports=24
    if product and backbone!='grid_dit' and model_spec is None:
        raise ValueError(f'{method} requires joint multiband input/output: use grid_dit or a custom product-state model, not independent per-band {backbone} calls')
    if backbone=='grid_dit':
        args=dict(channels=ports,width=width,depth=depth,num_heads=attention_heads,
                  patch=list(patch),position=position,max_stages=stages,heads=heads,
                  product_order='fine_first' if method in ('lapflow','mdm') else 'coarse_first',
                  causal_scales=method=='lapflow')
    elif backbone=='grid_conv':args=dict(channels=ports,width=width,heads=heads,max_stages=stages)
    elif backbone=='dit_grid':
        if first_frame and len(shape)==4 and any(a[0]!=b[0] for a,b in zip(grids[:-1],grids[1:])) and method=='pyramidal_wan_path':
            raise ValueError('PyramidalWan first/tail clocks need grid_dit or wan_grid; original DiT accepts only one clock')
        if tuple(patch)[0]!=1 or patch[-2]!=patch[-1]:raise ValueError('Original DiT uses a square spatial patch and temporal patch 1')
        if shape[-2]!=shape[-1]:raise ValueError('Original DiT initializes a square reference grid; use grid_dit for rectangular configs')
        reference_size=shape[-1]//8 if method=='dctdiff' else shape[-1]
        args=dict(architecture=dict(input_size=reference_size,patch_size=patch[-1],in_channels=ports,
                    hidden_size=width,depth=depth,num_heads=attention_heads,num_classes=1,class_dropout_prob=0.),
                  heads=heads,time_scale=1000. if method in STAGE_METHODS or method=='flow' else 1.)
    elif backbone=='wan_grid':
        args=dict(architecture=dict(patch_size=list(patch),in_channels=ports,out_channels=ports*(len(heads) if heads else 1),
                    num_attention_heads=attention_heads,attention_head_dim=width//attention_heads,
                    num_layers=depth,text_dim=16,freq_dim=32,ffn_dim=width*4),heads=heads,
                  time_scale=1000. if method in STAGE_METHODS or method=='flow' else 1.)
    else:
        if model_spec is None:raise ValueError('Unknown backbone; provide an explicit registry model specification')
        args={}
    model=copy.deepcopy(model_spec) if model_spec else {'name':backbone,'args':args}
    if separate_stages:model={'name':'stage_models','args':{'models':[copy.deepcopy(model) for _ in grids]}}
    config=dict(name=f'{method}_{"video" if len(shape)==4 else "image"}_{backbone}',image_shape=shape,model=model,
                training={'optimizer':{'lr':1e-4,'ema_decay':.999,'grad_clip':1.},'time_sampler':{'name':'uniform'}},
                evidence={'claim':'controlled_common_backbone_mathematical_recipe',
                          'not_a_paper_best_result_recipe':True,'no_asset_downloads':True,
                          'modality':'video' if len(shape)==4 else 'image',
                          'architecture_transplant':True,'source_method':method})
    if backbone=='wan_grid' and model_spec is None:
        config['training']['conditioning']={'text':{'kind':'field','field':'text_embeddings'}}
    if method in STAGE_METHODS:
        kwargs={'steps':steps,**copy.deepcopy(process_args or {})}
        if decoupled:kwargs['target']={'name':'rdd_decoupled'}
        if method=='pyramidal_wan_path' and len(grids)!=3 and 'natural_bounds' not in kwargs:
            raise ValueError('Paper default PyramidalWan path has three stages; specify natural_bounds for another count')
        config.update(task='multires',grids=grids,first_frame=first_frame,process={'name':method,'args':kwargs})
    elif method=='flow':
        config.update(path={'name':'flow','args':{'orientation':'data_to_noise'}},
                      target={'name':'joint','args':{'quantity':'velocity'}},clock={'name':'linear','args':{'start':1.,'end':0.}})
    elif method=='ddpm':
        config.update(task='multires',grids=[shape[1:]],process={'name':'pddpm','args':{'steps':steps,'train_steps':1000}})
        config['evidence']['sampling']='1000-step trained DDPM marginal; explicitly respaced ancestral inference'
    else:
        fixture='lapflow_2' if method=='lapflow' else method
        old=json.loads((Path(__file__).resolve().parents[1]/'configs'/'native_fixtures'/f'{fixture}.json').read_text())
        for k in ('batch','target','representation','clock','sampler','prediction_adapter'):
            if k in old:config[k]=old[k]
        config['training']['time_sampler']=old['training']['time_sampler']
        if method=='relay':
            config['model']={'name':'edm_grid','args':{'model':model,'img_resolution':shape[-1]}}
            config['batch']['args']['native_loss']=True
            config['sampler']['args'].update(steps=steps)
        elif method=='ssd':
            if shape[-1]!=shape[-2] or shape[-1]%2:raise ValueError('SSD source schedule requires square even grids')
            resolutions=[shape[-1]]*(steps//2)+[shape[-1]//2]*(steps-steps//2)
            if steps<2:raise ValueError('SSD requires at least two steps')
            betas=[.001+(.02-.001)*i/(steps-1) for i in range(steps)]
            kwargs=dict(resolutions=resolutions,betas=betas,unet_res_layer_map={str(r):r for r in set(resolutions)})
            config['model']={'name':'ssd_grid','args':{'model':model}}
            config['batch']['args']=copy.deepcopy(kwargs);config['sampler']['args']=copy.deepcopy(kwargs)
            config['training']['time_sampler']['args']['steps']=steps
        elif method=='dfm':config['clock']['args']={'steps':[steps,steps]}
        elif method=='lapflow':config['sampler']['args']['step_size']=1/steps
        elif method=='mdm':
            config['evidence']['excluded']='NestedUNet architecture and private data curriculum; joint extended-space objective is retained'
        elif method=='dctdiff':
            config['representation']['name']='dct_ycbcr_grid'
            config['representation']['args']['resolution']=shape[-1]
            config['sampler']['args']['steps']=max(3,steps)
    if inference_policy:
        if method!='flow':raise ValueError('Training-free grid policies currently require ordinary data-to-noise velocity flow')
        h,w=shape[-2:]
        if inference_policy=='speed':
            options=dict(resolutions=[[h//4,w//4],[h//2,w//2],[h,w]],steps=[steps]*3,boundaries=[1.,.6,.3,0.])
        elif inference_policy=='bottleneck':
            options=dict(resolutions=[[h,w],[h//2,w//2],[h,w]],steps=[steps]*3,start_indices=[0,steps//2,steps//2],shifts=[1.,1.,1.])
        else:raise ValueError('Unknown inference policy')
        options.update(inference_args or {})
        if options['resolutions'][-1]!=shape[-2:]:raise ValueError('Inference must finish at the configured output grid')
        config['sampler']={'name':inference_policy+'_grid','args':options}
        config['evidence']['inference_policy']=inference_policy+'_core_controlled_schedule'
        config['evidence']['training_objective']='ordinary flow; no claim of paper-specific fine-tuning'
    elif inference_args:raise ValueError('Inference arguments require an inference policy')
    # Check mathematical grid contracts before writing a runnable config.
    from .cli import make_method
    make_method(config)
    return config


def configure(args):
    shape=[int(x) for x in args.shape.split(',')]
    grids=json.loads(args.grids) if args.grids else None
    model=json.loads(Path(args.model_json).read_text()) if args.model_json else None
    config=recipe(args.method,shape,args.backbone,grids,args.width,args.depth,args.attention_heads,
                  tuple(int(x) for x in args.patch.split(',')),args.position,args.first_frame,
                  args.separate_stages,args.decoupled,model,json.loads(args.process_args),args.sampling_steps,
                  args.inference_policy,json.loads(args.inference_args))
    dest=Path(args.output);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():raise FileExistsError(f'Refusing to replace {dest}; choose a new config path')
    dest.write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'config':str(dest),'claim':config['evidence']['claim'],'shape':shape}))
