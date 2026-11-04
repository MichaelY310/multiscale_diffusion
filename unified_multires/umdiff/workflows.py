"""Offline inference-only workflows with explicit local asset ownership."""
from pathlib import Path
import numpy as np
import torch
from .registry import build,register
from .randomness import RandomStream

@register('model','flux_transformer')
def flux_transformer(**kwargs):
    from diffusers import FluxTransformer2DModel
    return FluxTransformer2DModel(**kwargs)

@register('model','qwen_image_transformer')
def qwen_transformer(**kwargs):
    from diffusers import QwenImageTransformer2DModel
    return QwenImageTransformer2DModel(**kwargs)

@register('model','qwen_image_vae')
def qwen_vae(**kwargs):
    from diffusers import AutoencoderKLQwenImage
    return AutoencoderKLQwenImage(**kwargs)

def local_module(spec,checkpoint,device,dtype,synthetic_fixture):
    if not synthetic_fixture and checkpoint is None:
        raise ValueError('A local checkpoint is required for every learned module outside an explicit synthetic fixture')
    module=build('model',spec)
    if checkpoint is not None:
        path=Path(checkpoint)
        if not path.is_file():raise FileNotFoundError(path)
        if path.suffix=='.safetensors':
            from safetensors.torch import load_file
            state=load_file(str(path))
        else:state=torch.load(path,map_location='cpu',weights_only=True)
        if isinstance(state,dict) and 'state_dict' in state:state=state['state_dict']
        module.load_state_dict(state,strict=True)
    return module.to(device=device,dtype=dtype).eval()

def run(config,embeddings,device='cpu',seed=0):
    """Each invocation owns its models. NAMI's vendored Diffusers is process-local.

    Input embeddings must already exist locally. This API never constructs a
    tokenizer/text encoder or calls from_pretrained / snapshot_download.
    """
    allowed={'name','model','vae','sr_model','bridge','checkpoints','sampler','initial_shape','dtype','synthetic_fixture','synthetic_embedding_shape','super_resolution','models','guidance'}
    if set(config)-allowed:raise ValueError(f'Unknown workflow fields: {set(config)-allowed}')
    torch.manual_seed(seed);np.random.seed(seed)
    dtype={'float32':torch.float32,'bfloat16':torch.bfloat16,'float16':torch.float16}[config.get('dtype','float32')]
    fixture=config.get('synthetic_fixture',False);weights=config.get('checkpoints',{})
    if config['name']=='relay':
        if set(config.get('models',{}))!={'low','high'} or set(weights)-{'low','high'}:raise ValueError('Relay requires low/high models and local checkpoints')
        models=torch.nn.ModuleDict({key:local_module(spec,weights.get(key),device,dtype,fixture) for key,spec in config['models'].items()})
        stream=RandomStream(seed,device);initial=stream.normal(torch.empty(*config['initial_shape'],device=device,dtype=dtype))
        sampler=build('sampler',{'name':'relay_chain','args':config.get('sampler',{})})
        return {'pixels':sampler(models,initial,stream,model_kwargs={k:v.to(device) for k,v in embeddings.items()})}
    if set(weights)-{'model','vae','sr_model','bridge'}:raise ValueError('Unknown workflow checkpoint key')
    model=local_module(config['model'],weights.get('model'),device,torch.float32 if config['name']=='speed_wan' else dtype,fixture)
    if config['name']=='speed_pixelgen':
        if config['vae']['name']!='native_pixelae':raise ValueError('Native PixelGen workflow requires its stateless PixelAE')
        vae=local_module(config['vae'],weights.get('vae'),device,dtype,True)
    else:vae=local_module(config['vae'],weights.get('vae'),device,torch.float32 if config['name']=='speed_wan' else dtype,fixture)
    from .tree import map_tree
    kwargs=map_tree(lambda value:value.to(device=device,dtype=dtype if value.is_floating_point() else value.dtype),embeddings)
    stream=RandomStream(seed,device)
    with torch.no_grad():
        if config['name'] in ('mrflow','mrflow_qwen'):
            from diffusers import FluxPipeline,FluxImg2ImgPipeline,QwenImagePipeline,QwenImageImg2ImgPipeline,FlowMatchEulerDiscreteScheduler
            from .superresolution import RealESRGANPixels,MrFlow
            sr=local_module(config['sr_model'],weights.get('sr_model'),device,torch.float32,fixture)
            is_qwen=config['name']=='mrflow_qwen'
            pipe=QwenImagePipeline(FlowMatchEulerDiscreteScheduler(),vae,None,None,model) if is_qwen else FluxPipeline(scheduler=FlowMatchEulerDiscreteScheduler(),vae=vae,transformer=model,text_encoder=None,text_encoder_2=None,tokenizer=None,tokenizer_2=None)
            pipe.set_progress_bar_config(disable=True)
            refiner=(QwenImageImg2ImgPipeline if is_qwen else FluxImg2ImgPipeline)(**pipe.components);refiner.set_progress_bar_config(disable=True)
            workflow=MrFlow(seed=seed,**config.get('sampler',{}))
            if is_qwen:
                pixels=workflow(pipe,RealESRGANPixels(sr,**config.get('super_resolution',{})),refiner,conditioning={
                    'prompt':None,'negative_prompt':None,**kwargs,'true_cfg_scale':config.get('guidance',4.)})
            else:pixels=workflow(pipe,RealESRGANPixels(sr,**config.get('super_resolution',{})),refiner,kwargs['prompt_embeds'],kwargs['pooled_prompt_embeds'])
            return {key:torch.from_numpy(np.asarray(value).copy()).permute(2,0,1)[None] for key,value in pixels.items()}
        initial=stream.normal(torch.empty(*config['initial_shape'],device=device,dtype=dtype))
        if config['name']=='nami':
            from .boundaries import LearnedAffine
            bridge=LearnedAffine(**config.get('bridge',{}))
            if weights.get('bridge'):
                bridge.load_state_dict(torch.load(weights['bridge'],map_location='cpu',weights_only=True),strict=True)
            elif not fixture:raise ValueError('NAMI requires a local learned BridgeFlow checkpoint')
            kwargs['bridge_flow']=bridge.to(device=device,dtype=dtype).eval()
            sampler=build('sampler',{'name':'nami','args':config.get('sampler',{})})
        elif config['name']=='speed':
            kwargs.setdefault('text_ids',torch.zeros(kwargs['prompt_embeds'].shape[1],3,device=device,dtype=dtype))
            sampler=build('sampler',{'name':'speed_flux','args':{**config.get('sampler',{}),'seed':seed}})
        elif config['name'] in ('speed_pixelgen','speed_wan'):
            sampler=build('sampler',{'name':config['name'],'args':{**config.get('sampler',{}),'seed':seed}})
        else:raise ValueError(f'Unknown offline workflow {config["name"]}')
        latents=sampler(model,initial,stream=stream,model_kwargs=kwargs)
        if config['name']=='speed_pixelgen':return {'latents':latents,'pixels':vae.decode(latents)}
        if config['name']=='speed_wan':return {'latents':latents,'pixels':vae.decode(latents[None].float(),return_dict=False)[0]}
        pixels=vae.decode(latents/vae.config.scaling_factor+vae.config.shift_factor,return_dict=False)[0]
        return {'latents':latents,'pixels':pixels}
