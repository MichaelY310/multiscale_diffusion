#!/usr/bin/env python3
"""Data, optimizer and two-optimizer accumulation comparisons with native code."""
import os,sys,json,random,traceback,inspect
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from umdiff import build,native
from umdiff.engine import Method
from umdiff.training import Trainer,LocalImages
from umdiff.randomness import GlobalRandomStream
from umdiff.input_transforms import VAEMoments
import conformance,validate_training as training
from activation_trace import ActivationAudit
C=conformance.compare;OUT=ROOT/'evidence/recipe_extensions';RUN=ROOT/'runs/recipe_extensions/runtime';RUN.mkdir(parents=True,exist_ok=True)
def optimization():
    model=build('model',{'name':'tiny_conv','args':{'channels':3,'width':8}});method=Method({'path':{'name':'flow'}});a=deepcopy(model);ema=deepcopy(model)
    source=native.extract('dctdiff/utils.py',['customized_lr_scheduler','ema'],dict(nn=torch.nn));oa=torch.optim.AdamW(a.parameters(),lr=.0002,betas=(.99,.99),weight_decay=.03)
    scheduler=source['customized_lr_scheduler'](oa,200)
    recipe=dict(lr=.0002,betas=(.99,.99),weight_decay=.03,ema_decay=.9999,ema_style='scaled_add',lr_schedule={'name':'warmup','warmup_steps':200,'offset':0},accumulation_steps=2,randomness='native_global')
    trainer=Trainer(method,deepcopy(model),**recipe)
    inputs=torch.randn(4,3,8,8);t=torch.rand(4);noise=torch.randn_like(inputs)
    for step in range(128):
        oa.zero_grad(set_to_none=True)
        for i in range(2):method.training(a,inputs[i*2:(i+1)*2],t[i*2:(i+1)*2],GlobalRandomStream(),noise=noise[i*2:(i+1)*2]).loss.mean().div(2).backward()
        expected_lr=oa.param_groups[0]['lr'];oa.step();source['ema'](ema,a,.9999);scheduler.step();trainer.step(inputs,t,noise=noise)
        C('runtime/LR/'+str(step),trainer.optimizer.param_groups[0]['lr'],expected_lr,atol=0,rtol=0)
        for key,expected in a.state_dict().items():C('runtime/weights/'+str(step)+'/'+key,trainer.model.state_dict()[key],expected,atol=0,rtol=0)
        for key,expected in ema.state_dict().items():C('runtime/EMA/'+str(step)+'/'+key,trainer.ema.state_dict()[key],expected,atol=0,rtol=0)
        if step==63:
            trainer.save(RUN/'resume.pt');restored=Trainer(method,deepcopy(model),**recipe);restored.load(RUN/'resume.pt');trainer=restored
    return {'updates':128,'microbatches_per_update':2,'resume_after':64,'scope':'Native DCT/DiMR LR and EMA arithmetic, independent AdamW; shared controlled objective for this optimizer-only check.'}
def data():
    from torchvision import transforms
    directory=RUN/'images';directory.mkdir(exist_ok=True);Image.fromarray(np.random.default_rng(32).integers(0,256,(97,111,3),dtype=np.uint8)).save(directory/'image.png')
    crop=native.extract('pixelflow/pixelflow/data_in1k.py',['center_crop_arr'],dict(Image=Image,np=np))['center_crop_arr'];checks=0
    for mode in ('adm_center','pixelflow_random'):
        actual=LocalImages(directory,32,crop=mode,horizontal_flip=True,flip_rng='torch',normalization='torchvision')
        transform=transforms.Compose(([transforms.Lambda(lambda image:crop(image,32))] if mode=='adm_center' else [transforms.Resize(36,interpolation=transforms.InterpolationMode.LANCZOS),transforms.RandomCrop(32)])+[transforms.RandomHorizontalFlip(),transforms.ToTensor(),transforms.Normalize([.5]*3,[.5]*3,inplace=True)])
        for seed in range(16):
            training.seed(seed);expected=transform(Image.open(directory/'image.png').convert('RGB'));training.seed(seed);C('data/'+mode+'/'+str(seed),actual[0],expected,atol=0,rtol=0);checks+=1
    moments=torch.randn(3,8,8,8);sample=native.method('dimr/libs/autoencoder.py','sample',dict(torch=torch))
    training.seed(92);expected=sample(SimpleNamespace(scale_factor=.18215),moments);training.seed(92);C('data/native_VAE_moments',VAEMoments()(moments,GlobalRandomStream()),expected,atol=0,rtol=0)
    return {'image_transform_cases':checks,'moment_sampling':'native clamp, RNG, arithmetic exactly equal'}
def rmd_accumulation():
    import validate_rmd as reference
    from umdiff.distillation import RMDTrainer
    src,loss,env,cfg,p,objective,model,template,text,clock,high=reference.fixtures(1)
    a=deepcopy(model);fake=deepcopy(model);real=deepcopy(model).eval().requires_grad_(False)
    config={'task':'rmd','image_shape':[4,2,4,6],'distillation':{**vars(cfg),'target_size':[8,12]},'training':{'accumulation_steps':2}}
    config['distillation'].pop('gan');trainer=RMDTrainer(config,deepcopy(model),p.uncond_prompt_embeds if hasattr(p,'uncond_prompt_embeds') else objective.predictor.negative)
    oa=torch.optim.AdamW(a.parameters(),lr=5e-6,betas=(0.,.95),weight_decay=1e-4);of=torch.optim.AdamW(fake.parameters(),lr=2e-5,betas=(0.,.95),weight_decay=1e-4)
    template=template.repeat(2,1,1,1,1);text=text.repeat(2,1,1);audit=ActivationAudit('rmd_accumulation',[('generator',a,trainer.model),('fake',fake,trainer.fake),('teacher',real,trainer.real)],OUT/'activation')
    for step in range(128):
        audit.begin(step);audit.side('native');training.seed(7300+step);oa.zero_grad(set_to_none=True);of.zero_grad(set_to_none=True);a.train();fake.train()
        for micro in range(2):
            inputs=template[micro*2:(micro+1)*2];condition=text[micro*2:(micro+1)*2];torch.randn(inputs.shape);noise=torch.randn_like(inputs);a.eval()
            with torch.no_grad():clean,trajectory=p.generate_new_upsample(a,None,noise,noise,condition,None,steps=6,eta=.5,return_mid=True,flow_shift_trans=1,relusion_shift=2,target_size=(8,12))
            trajectory.reverse();indices=torch.randint(4,6,(2,));example=torch.randn_like(clean[int(indices[0])])
            for j in range(2):example[j]=clean[int(indices[j])][j]
            a.train()
            for phase,opt,network in ((1,of,fake),(2,oa,a)):
                args=(cfg,p,a,fake)+((real,) if phase==2 else ())+(clock,high,example,trajectory,condition,None,6,4,objective.scales,1000,torch.float32,'cpu')
                if phase==2:args+=(None,)
                args+=(step,);value=env['_phase'+str(phase)](*args,target_size=(8,12))/2;value.backward()
                if micro==1:
                    if phase==2:torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
                    opt.step();opt.zero_grad(set_to_none=True)
        audit.side('unified');training.seed(7300+step);trainer.step(template,text);audit.finish_segment()
        for prefix,x,y in [('generator',a,trainer.model),('fake',fake,trainer.fake)]:
            for key,v in x.state_dict().items():C('rmd_accum/'+str(step)+'/'+prefix+'/'+key,y.state_dict()[key],v,atol=0,rtol=0)
    record=audit.close()
    if record['structural_mismatches'] or any(s['pairs']!=s['within_tolerance'] for s in record['totals'].values()):raise AssertionError('RMD accumulation activation audit failed')
    return record
def main():
    torch.set_num_threads(2);training.seed(20261009);failure=None;details={}
    try:details={'optimization':optimization(),'data':data(),'rmd_accumulation':rmd_accumulation()}
    except Exception:failure=traceback.format_exc();print(failure,flush=True)
    report={'status':'failed' if failure else 'passed','failure':failure,'device':'cpu','gpu_used':False,'checks':conformance.CHECKS,'details':details,'source_executions':native.AUDIT}
    (OUT/'runtime.json').write_text(json.dumps(report,indent=2));print(json.dumps({'status':report['status'],'failure':failure}));return bool(failure)
if __name__=='__main__':raise SystemExit(main())
