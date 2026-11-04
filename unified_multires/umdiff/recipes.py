"""Offline initialization and explicit optimization recipes, outside path math."""
from pathlib import Path
import fnmatch,hashlib,math
import torch

def state_file(path,key=None,prefix=None):
    path=Path(path)
    if not path.is_file():raise FileNotFoundError(f'Local asset required: {path}')
    if path.suffix=='.safetensors':
        from safetensors.torch import load_file
        value=load_file(str(path))
    else:value=torch.load(path,map_location='cpu',weights_only=True)
    if key is not None:value=value[key]
    if not isinstance(value,dict) or not all(torch.is_tensor(v) for v in value.values()):raise ValueError('Select a tensor state dictionary explicitly with key')
    if prefix is not None:value={k[len(prefix):]:v for k,v in value.items() if k.startswith(prefix)}
    if not value:raise ValueError('Empty checkpoint selection')
    return value

def initialize(model,spec):
    spec=dict(spec);mode=spec.pop('mode','state_dict');path=spec.pop('path');key=spec.pop('key',None);prefix=spec.pop('prefix',None)
    core=model.model if hasattr(model,'model') else model
    submodule=spec.pop('submodule',None)
    if submodule:core=core.get_submodule(submodule)
    strict=spec.pop('strict',True)
    if spec:raise ValueError(f'Unknown initialization fields: {set(spec)}')
    value=state_file(path,key,prefix)
    if mode=='ppflow_sit':
        destination=core.state_dict();mapped={k:v for k,v in value.items() if k in destination}
        for suffix,repeat in [('1',4)]+([('2',2)] if core.patch_level==3 else []):
            mapped[f'x_embedder.proj{suffix}.weight']=(value['x_embedder.proj.weight']/float(repeat)).repeat_interleave(repeat,dim=1)
            mapped[f'x_embedder.proj{suffix}.bias']=value['x_embedder.proj.bias']
            mapped[f'final_layer.linear{suffix}.weight']=value['final_layer.linear.weight'].repeat_interleave(repeat,dim=0)
            mapped[f'final_layer.linear{suffix}.bias']=value['final_layer.linear.bias'].repeat_interleave(repeat,dim=0)
        # Stage embeddings are intentionally new, exactly like author strict=False.
        result=core.load_state_dict(mapped,strict=False)
    elif mode=='state_dict':result=core.load_state_dict(value,strict=strict)
    else:raise ValueError(f'Unknown initialization mapping {mode}')
    return {'path':str(Path(path).resolve()),'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),'mode':mode,'missing':result.missing_keys,'unexpected':result.unexpected_keys}

def freeze_parameters(model,patterns):
    selected=[]
    for pattern in patterns:
        matches=[(k,p) for k,p in model.named_parameters() if fnmatch.fnmatchcase(k,pattern)]
        if not matches:raise ValueError(f'Freeze pattern matches no parameters: {pattern}')
        for key,p in matches:p.requires_grad_(False);selected.append(key)
    return selected

def lr_factor(step,spec,images_seen=0):
    kind=spec.get('name','constant');warmup=int(spec.get('warmup_steps',0));offset=spec.get('offset',1)
    if kind=='image_warmup':return min(images_seen/max(spec['warmup_images'],1e-8),1.)
    if kind=='cosine_epoch':
        if spec.get('steps_per_epoch',0)<1 or spec.get('epochs',0)<1:raise ValueError('Epoch cosine needs positive steps_per_epoch and epochs')
        epoch=step//spec['steps_per_epoch'];floor=spec.get('minimum_ratio',0.)
        return floor+(1-floor)*.5*(1+math.cos(math.pi*min(epoch,spec['epochs'])/spec['epochs']))
    if warmup and step<warmup:return max(spec.get('minimum_step',0),step+offset)/warmup
    if kind in ('constant','warmup'):return 1.
    total=spec.get('total_steps')
    if total is None or total<=warmup:raise ValueError('Decay schedule requires total_steps > warmup_steps')
    progress=min(1.,max(0.,(step-warmup)/(total-warmup)));floor=spec.get('minimum_ratio',0.)
    if kind=='cosine':return floor+(1-floor)*.5*(1+math.cos(math.pi*progress))
    if kind=='linear':return floor+(1-floor)*(1-progress)
    raise ValueError(f'Unknown LR schedule {kind}')
