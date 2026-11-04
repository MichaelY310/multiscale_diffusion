"""Source backbones for offline SPEED routes; CPU substitutions stay explicit."""
import ast,hashlib,math,types,functools
import torch
from .. import native
from ..registry import register
from ..native_models import activate


@register('model','native_pixelgen')
def pixelgen(trained_grid=32,**kwargs):
    activate('pixelgen',['src'])
    from src.models.transformer.JiT_T2I import JiT_T2I
    from src.models.layers.rope import precompute_freqs_cis_ex2d
    from .speed_routes import attach_pixelgen_rope
    model=JiT_T2I(**kwargs);attach_pixelgen_rope(model,trained_grid,precompute_freqs_cis_ex2d)
    return model


@register('model','native_pixelae')
def pixelae(**kwargs):
    activate('pixelgen',['src'])
    from src.models.autoencoder.pixel import PixelAE
    return PixelAE(**kwargs)


@register('model','wan_video_vae')
def wan_vae(**kwargs):
    from diffusers import AutoencoderKLWan
    return AutoencoderKLWan(**kwargs)


@register('model','native_wan21_cpu')
def wan_cpu(**kwargs):
    from diffusers.configuration_utils import ConfigMixin,register_to_config
    from diffusers.models.modeling_utils import ModelMixin
    class CPUAMP:
        @staticmethod
        def autocast(dtype=None,enabled=True):return torch.autocast('cpu',dtype=torch.bfloat16,enabled=enabled and dtype!=torch.float32)
    def attention(q,k,v,k_lens=None,**options):
        q=q.to(v.dtype);k=k.to(v.dtype);mask=None
        if k_lens is not None:mask=torch.arange(k.shape[1])[None,None,None,:]<k_lens[:,None,None,None]
        return torch.nn.functional.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),attn_mask=mask).transpose(1,2)
    path=native.ROOT/'sources/wan21/wan/modules/model.py'
    names=[n.name for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
    env=native.extract('wan21/wan/modules/model.py',names,dict(torch=torch,nn=torch.nn,math=math,amp=CPUAMP,
        flash_attention=attention,ConfigMixin=ConfigMixin,register_to_config=register_to_config,ModelMixin=ModelMixin,
        T5_CONTEXT_TOKEN_NUMBER=512,FIRST_LAST_FRAME_CONTEXT_TOKEN_NUMBER=514))
    env.update(native.statements('wan21/wan/modules/model.py',232,235,env))
    model=env['WanModel'](**kwargs);forward=model.forward
    @functools.wraps(forward.__func__)
    def cpu_forward(self,*args,**kw):
        if next(self.parameters()).device.type!='cpu':raise RuntimeError('native_wan21_cpu explicitly substitutes CPU kernels and cannot certify CUDA')
        with torch.autocast('cpu',dtype=torch.bfloat16):return forward(*args,**kw)
    model.forward=types.MethodType(cpu_forward,model)
    model.eval().requires_grad_(False)
    model.compatibility_evidence=dict(source=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        substitutions=['Native CUDA AMP FP32 scopes translated to CPU AMP','Native FlashAttention substituted by masked CPU SDPA','CPU BF16 outer model autocast'],cuda_certified=False)
    return model
