"""CPU preflight of full recipes with small depth/width, plus legacy RDD parity."""
import ast
import copy
import gc
import hashlib
import json
import random
import time
import traceback
from pathlib import Path
import torch

from umdiff import build
from umdiff.cli import make_method, setup
from umdiff.experiment import load, resolve
from umdiff.randomness import RandomStream, GlobalRandomStream
from umdiff.rdd_native import diffusion_backend
from umdiff.tree import map_tree

ROOT=Path(__file__).resolve().parents[1]
CONFIGS=ROOT/'configs/comparison_imagenet256'
OUT=ROOT/'evidence/main_comparison_20261013'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(2)


def shrink_model(spec):
    spec=copy.deepcopy(spec)
    if spec['name']=='native_rdd_unified':spec['args'].update(hidden_size=32,depth=2,num_heads=4)
    elif spec['name']=='comparison_dit':spec['args']['architecture'].update(hidden_size=32,depth=2,num_heads=4)
    elif spec['name']=='ssd_grid':spec['args']['model']=shrink_model(spec['args']['model'])
    else:raise ValueError(spec['name'])
    return spec


def small(spec):
    c=resolve(spec)
    c['model']=shrink_model(c['model'])
    c['training'].pop('input_transform')
    c['training'].pop('input_transform_scope')
    c['training']['optimizer']['accumulation_steps']=2
    return c


def draw_initial(method,c,stream):
    shape=c['image_shape'].copy()
    name=c.get('sampler',{}).get('name')
    if hasattr(method,'initial_shape'):shape=method.initial_shape()
    elif name=='rdd_native_stages':shape[-2:]=[c['sampler']['args']['resolutions'][-1]]*2
    elif name=='ssd':shape[-2:]=[c['sampler']['args']['resolutions'][-1]]*2
    value=stream.normal(torch.empty(1,*shape))
    if getattr(method,'batch',None) is not None and name is None and hasattr(method.clock,'schedule'):
        value=dict(enumerate(stream.normal(v) for v in method.representation.encode(value)))
    return value


def preflight(path):
    spec=load(path);c=small(spec)
    method,net,trainer=setup(c,torch.device('cpu'),27)
    sampler=build('time_sampler',c['training']['time_sampler']);trainer.time_sampler=sampler
    images=torch.randn(2,4,32,32);labels=torch.tensor([11,47]);losses=[]
    for _ in range(2):
        result=trainer.step(images,None,model_kwargs={'y':labels},time_sampler=sampler)
        assert torch.isfinite(result.loss).all()
        losses.append(result.loss.mean().item())
    net=trainer.ema.eval();stream=RandomStream(71)
    initial=draw_initial(method,c,stream);calls=[0]
    # Native RDD invokes net.forward directly, bypassing root hooks. The first
    # qkv module is evaluated once in every scalar or joint-band model call.
    counter=next(m for k,m in net.named_modules() if k.endswith('blocks.0.attn.qkv'))
    hook=counter.register_forward_hook(lambda *args:calls.__setitem__(0,calls[0]+1))
    with torch.no_grad():
        result=method.sample(net,initial,steps=spec['sampling'].get('sample_cli_steps',20),stream=stream,model_kwargs={'y':labels[:1]})
    hook.remove()
    assert torch.is_tensor(result) and result.shape==(1,4,32,32) and torch.isfinite(result).all()
    row={'id':spec['id'],'suite':spec['suite'],'status':'passed','losses':losses,'sampling_forward_calls':calls[0],
         'sample_shape':list(result.shape),'sample_sha256':hashlib.sha256(result.contiguous().numpy().tobytes()).hexdigest(),
         'config_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
         'scope':'Original32/16/8/etc grids,1000-step tables and configured solver budgets; width32/depth2; synthetic model-domain input; no pretrained codec.'}
    del trainer,net,method,result;gc.collect()
    return row


def capture(model):
    values={};gradients={};counts={};handles=[]
    def register(name):
        def hook(module,args,out):
            index=counts.get(name,0);counts[name]=index+1
            def take(value,path=''):
                if torch.is_tensor(value):
                    key=f'{name}:{index}:{path}';values[key]=value.detach().clone()
                    if value.requires_grad:value.register_hook(lambda g,k=key:gradients.__setitem__(k,g.detach().clone()))
                elif isinstance(value,(list,tuple)):
                    for i,v in enumerate(value):take(v,path+'/'+str(i))
                elif isinstance(value,dict):
                    for k,v in value.items():take(v,path+'/'+str(k))
            take(out)
        return hook
    for name,module in model.named_modules():handles.append(module.register_forward_hook(register(name)))
    return values,gradients,handles


def exact(left,right):
    assert left.keys()==right.keys(),(left.keys()-right.keys(),right.keys()-left.keys())
    maximum=0.
    for key,a in left.items():
        b=right[key];assert a.dtype==b.dtype and a.shape==b.shape,key
        assert torch.equal(a,b),f'{key}: max_abs={(a-b).abs().max().item()}'
        assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8)),key
    return len(left)


def rdd_parity():
    spec=load(CONFIGS/'main/rdd_ddpm_xl_legacy.json');c=small(spec)
    method=make_method(c);model=build('model',c['model'])
    # Nonzero gates and readouts make the entire backward comparison informative.
    with torch.no_grad():
        for p in model.parameters():p.normal_(0,.03)
    original=copy.deepcopy(model)
    x=torch.randn(2,4,32,32);y=torch.tensor([3,19])
    backend=diffusion_backend();bounds=[0,300,600,1000]
    source_path=ROOT.parent/'2d/train_RDD_imagenet256_unified.py'
    tree=ast.parse(source_path.read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='downsample_latent')
    namespace={}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(source_path),'exec'),namespace)
    diffusions=[backend.create_diffusion_with_drift('',diffusion_steps=1000,c=i if i<2 else None,T_c=bounds[i],T_c_plus_1=bounds[i+1],sample_scale=2) for i in range(3)]
    aa,ag,ah=capture(original);ba,bg,bh=capture(model)
    torch.manual_seed(619);loss=0.
    for i,d in enumerate(diffusions):
        value=x if i==0 else namespace['downsample_latent'](x.detach(),32//2**i)
        t=torch.randint(0,d.num_timesteps,(len(value),))
        loss=loss+d.training_losses(original,value,t,model_kwargs={'y':y})['loss'].mean()
    loss.backward()
    torch.manual_seed(619);got=method.training(model,x,None,GlobalRandomStream(),model_kwargs={'y':y}).loss
    got.backward()
    assert torch.equal(loss,got),(loss,got)
    counts={'module_outputs':exact(aa,ba),'activation_gradients':exact(ag,bg),
            'parameter_gradients':exact({k:p.grad for k,p in original.named_parameters() if p.grad is not None},{k:p.grad for k,p in model.named_parameters() if p.grad is not None})}
    for h in ah+bh:h.remove()
    return {'status':'passed','loss':loss.item(),'bitwise_equal':True,'counts':counts,
            'scope':'Unchanged original DiTMultiRes class; two blocks,width32; full32/16/8 grids, exact1000-step linear schedule and legacy three-stage training objective; common nonzero weights/input/RNG.'}


report={'device':'cpu','formal_training_started':False,'configs':[]}
for row in json.loads((CONFIGS/'manifest.json').read_text())['entries']:
    path=CONFIGS/row['file'];start=time.monotonic()
    try:result=preflight(path)
    except Exception:
        result={'id':row['id'],'suite':row['suite'],'status':'failed','traceback':traceback.format_exc()}
    result['seconds']=time.monotonic()-start;report['configs'].append(result)
    (OUT/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(result),flush=True)
try:report['legacy_rdd_parity']=rdd_parity()
except Exception:report['legacy_rdd_parity']={'status':'failed','traceback':traceback.format_exc()}
report['cuda_initialized']=torch.cuda.is_initialized()
report['status']='passed' if all(r['status']=='passed' for r in report['configs']) and report['legacy_rdd_parity']['status']=='passed' and not report['cuda_initialized'] else 'failed'
(OUT/'PREFLIGHT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'status':report['status'],'rdd':report['legacy_rdd_parity']}),flush=True)
raise SystemExit(0 if report['status']=='passed' else 1)
