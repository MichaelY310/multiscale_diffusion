"""Author backbones, instantiated from source with no pretrained downloads.

Repositories with colliding top-level names run in separate processes. Refuse
an import collision instead of silently importing another paper's `models`.
"""
import importlib,sys,types
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from . import native
from .locations import rdd_root
from .registry import register

def activate(repo,names):
    root=native.ROOT/'sources'/repo
    for name in names:
        module=sys.modules.get(name)
        if module is not None:
            file=getattr(module,'__file__',None)
            if file is not None and not Path(file).resolve().is_relative_to(root.resolve()):
                raise RuntimeError(f'Native repository namespace collision: {name}. Run each native backbone in its own process.')
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    return root

def namespace(value):
    if isinstance(value,dict):return SimpleNamespace(**{k:namespace(v) for k,v in value.items()})
    if isinstance(value,list):return [namespace(v) for v in value]
    return value

@register('model','native_dctdiff')
def dctdiff(**kwargs):
    activate('dctdiff',['libs'])
    return importlib.import_module('libs.uvit').UViT(**kwargs)

@register('model','native_rdd')
def rdd(**kwargs):
    root=rdd_root()/'DiT'
    spec=importlib.util.spec_from_file_location('umdiff_rdd_dit',root/'models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.DiT(**kwargs)

@register('model','native_rrdb')
def rrdb(**kwargs):
    activate('realesrgan',['RealESRGAN'])
    import huggingface_hub
    if not hasattr(huggingface_hub,'cached_download'):
        def blocked(*a,**kw):raise RuntimeError('Downloads are disabled; use locally supplied SR state dict')
        huggingface_hub.cached_download=blocked
    return importlib.import_module('RealESRGAN.rrdbnet_arch').RRDBNet(**kwargs)

@register('model','native_nami')
def nami(**kwargs):
    activate('nami',['diffusers'])
    import transformers.utils
    if not hasattr(transformers.utils,'FLAX_WEIGHTS_NAME'):transformers.utils.FLAX_WEIGHTS_NAME='flax_model.msgpack'
    from diffusers import FluxTransformer2DModel
    return FluxTransformer2DModel(**kwargs)

@register('model','native_dimr')
def dimr(config):
    import os
    os.environ.setdefault('K_DIFFUSION_USE_COMPILE','0');os.environ.setdefault('USE_FLASH_2','0')
    activate('dimr',['libs'])
    return importlib.import_module('libs.core.blocks').MRModel(namespace(config))

@register('model','native_ssd')
def ssd(**kwargs):
    activate('ssd',['guided_diffusion'])
    if 'unet_res_layer_map' in kwargs:kwargs['unet_res_layer_map']={int(k):v for k,v in kwargs['unet_res_layer_map'].items()}
    return importlib.import_module('guided_diffusion.unet').FlexiUNet(**kwargs)

@register('model','native_dfm')
def dfm(**kwargs):
    activate('dfm',['models'])
    return importlib.import_module('models.sit_dfm').DFMSiT(**kwargs)

@register('model','native_lapflow')
def lapflow(**kwargs):
    activate('lapflow',['models'])
    return importlib.import_module('models.MultiScaleDiT').MultiScaleDiT(**kwargs)

@register('model','native_ppflow')
def ppflow(**kwargs):
    activate('ppflow',['models','modules'])
    return importlib.import_module('models').PPFlow(**kwargs)

def sdpa_varlen(q,k,v,cu_seqlens_q,cu_seqlens_k,max_seqlen_q,max_seqlen_k,dropout_p=0.,softmax_scale=None,causal=False,**kwargs):
    """Exact attention equation; CPU fallback for native packed FlashAttention.

    The source model already upcasts q/k/v to fp32. Kernel rounding may differ
    from GPU FlashAttention; this fallback must be reported explicitly.
    """
    outputs=[]
    for i in range(len(cu_seqlens_q)-1):
        qs,qe=int(cu_seqlens_q[i]),int(cu_seqlens_q[i+1]);ks,ke=int(cu_seqlens_k[i]),int(cu_seqlens_k[i+1])
        output=torch.nn.functional.scaled_dot_product_attention(q[qs:qe].transpose(0,1)[None],k[ks:ke].transpose(0,1)[None],v[ks:ke].transpose(0,1)[None],dropout_p=dropout_p,is_causal=causal,scale=softmax_scale)
        outputs.append(output[0].transpose(0,1))
    return torch.cat(outputs)

@register('model','native_pixelflow')
def pixelflow(cpu_attention=True,**kwargs):
    activate('pixelflow',['pixelflow'])
    module=importlib.import_module('pixelflow.model')
    if cpu_attention:module.flash_attn_varlen_func=sdpa_varlen
    return module.PixelFlowModel(**kwargs)

@register('model','native_relay')
def relay(**kwargs):
    # Persistence affects source pickling, not forward/backward math. State-dict
    # checkpoints remain native-compatible. No xformers call for eff_attn=False.
    from einops import rearrange
    names=['weight_init','Linear','Conv2d','GroupNorm','AttentionOp','get_xformers_flash_attention_op','UNetBlock','PositionalEmbedding','FourierEmbedding','SongUNet','DhariwalUNet','EDMPrecond']
    env=native.extract('relay/training/networks.py',names,dict(np=np,torch=torch,silu=torch.nn.functional.silu,rearrange=rearrange,persistence=SimpleNamespace(persistent_class=lambda cls:cls)))
    if kwargs.get('eff_attn',False):raise ValueError('CPU native adapter requires eff_attn=False; use the full source module for xformers')
    return env['EDMPrecond'](**kwargs)

def import_mdm():
    """Import the released classes without constructing a network or drawing RNG."""
    root=native.ROOT/'sources/mdm/ml-mdm-matryoshka'
    if str(root) not in sys.path:sys.path.insert(0,str(root))
    # Unused CLI reader (mlx.data) and S3 download interfaces are unavailable in
    # the CPU fixture. Make every attempted use fail, never replace model math.
    def blocked(*a,**k):raise RuntimeError('Model/data downloads are disabled; supply local assets explicitly')
    if 'ml_mdm.reader' not in sys.modules:
        reader=types.ModuleType('ml_mdm.reader');reader.ReaderConfig=blocked
        sys.modules['ml_mdm.reader']=reader
    if 'torchinfo' not in sys.modules and importlib.util.find_spec('torchinfo') is None:
        info=types.ModuleType('torchinfo');info.summary=blocked;sys.modules['torchinfo']=info
    if 'ml_mdm.s3_helpers' not in sys.modules:
        s3=types.ModuleType('ml_mdm.s3_helpers');s3.download_object_from_full_path=blocked;sys.modules['ml_mdm.s3_helpers']=s3
    from ml_mdm.models.unet import UNetConfig,ResNetConfig
    from ml_mdm.models.nested_unet import NestedUNet,NestedUNetConfig
    return UNetConfig,ResNetConfig,NestedUNet,NestedUNetConfig

@register('model','native_mdm')
def mdm(input_channels=4,output_channels=4,config=None):
    UNetConfig,ResNetConfig,NestedUNet,NestedUNetConfig=import_mdm()
    def make(spec,nested=True):
        spec=dict(spec)
        if isinstance(spec.get('resnet_config'),dict):spec['resnet_config']=ResNetConfig(**spec['resnet_config'])
        if 'inner_config' in spec:spec['inner_config']=make(spec['inner_config'],'inner_config' in spec['inner_config'])
        cls=NestedUNetConfig if nested else UNetConfig
        return cls(**spec)
    if config is None:
        config={'resolution_channels':'32,32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,
                'inner_config':{'resolution_channels':'32','attention_levels':'','num_resnets_per_resolution':'1','conditioning_feature_dim':8,'nesting':True}}
    cfg=make(config)
    initial=getattr(cfg,'initialize_inner_with_pretrained',None)
    if initial is not None:
        if not Path(initial).is_file():raise FileNotFoundError('MDM inner initialization requires an existing local checkpoint')
        cfg.initialize_inner_with_pretrained=None
    network=NestedUNet(input_channels,output_channels,cfg)
    if initial is not None:
        from .recipes import state_file
        value=state_file(initial,key='state_dict')
        # Strict loading avoids the source's caught-error fallback to random weights.
        network.inner_unet.load_state_dict(value,strict=True)
    return network
