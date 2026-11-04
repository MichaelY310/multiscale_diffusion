#!/usr/bin/env python3
"""Exercise real local tensor/image inputs, conditional sampling and offline assets."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import numpy as np
import torch
from validate_cli import equal

def workflow_case(name):
    from umdiff.workflows import run,local_module
    from umdiff.boundaries import LearnedAffine
    config=json.loads((ROOT/'configs/inference_only'/f'{name}.json').read_text())
    directory=ROOT/'runs'/'offline_entrypoints'/name;directory.mkdir(parents=True,exist_ok=True)
    shape=config['synthetic_embedding_shape'];g=torch.Generator().manual_seed(77)
    embeddings={'prompt_embeds':torch.randn(shape,generator=g),'pooled_prompt_embeds':torch.randn(shape[0],shape[-1],generator=g)}
    if name=='nami':embeddings.update(negative_prompt_embeds=torch.randn(shape,generator=g),negative_pooled_prompt_embeds=torch.randn(shape[0],shape[-1],generator=g))
    expected=run(config,embeddings,seed=42)
    torch.manual_seed(42);weights={};dtype=getattr(torch,config.get('dtype','float32'))
    for key in ('model','vae','sr_model'):
        if key not in config:continue
        module=local_module(config[key],None,'cpu',torch.float32 if key=='sr_model' else dtype,True)
        path=directory/(key+'.pt');torch.save(module.state_dict(),path);weights[key]=str(path)
    if name=='nami':
        path=directory/'bridge.pt';torch.save(LearnedAffine().state_dict(),path);weights['bridge']=str(path)
    config.update(synthetic_fixture=False,checkpoints=weights)
    actual=run(config,embeddings,seed=42)
    if not equal(actual,expected):raise AssertionError('Local state-dict reload differs from the common initialized modules')
    torch.save(embeddings,directory/'embeddings.pt');(directory/'config.json').write_text(json.dumps(config,indent=2))
    return dict(name=name,status='passed',local_weight_reload_bitwise=True,output_shapes={k:list(v.shape) for k,v in actual.items()})

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workflow',choices=['nami','speed','mrflow']);args=parser.parse_args();torch.set_num_threads(2)
    if args.workflow:
        result=workflow_case(args.workflow);(ROOT/'evidence'/f'{args.workflow}_offline_assets.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));return
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'_deps')]))
    directory=ROOT/'runs'/'offline_entrypoints';directory.mkdir(parents=True,exist_ok=True);cases=[]
    def command(argv):
        result=subprocess.run([sys.executable,*argv],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        cases.append(dict(argv=argv,exit_code=result.returncode,log=result.stdout[-4000:]))
        (ROOT/'evidence'/'offline_entrypoints.json').write_text(json.dumps({'device':'cpu','gpu_used':False,'cases':cases},indent=2))
        if result.returncode:raise RuntimeError(result.stdout)
    # All inputs are created here. No external images or dataset are requested.
    from PIL import Image
    generator=torch.Generator().manual_seed(91);images=torch.rand(19,3,32,32,generator=generator)*2-1
    torch.save({'images':images,'labels':torch.arange(19)%2},directory/'images.pt')
    folder=directory/'images';folder.mkdir(exist_ok=True)
    for i,image in enumerate(images):
        path=folder/str(i%2);path.mkdir(exist_ok=True)
        Image.fromarray(((image+1)*127.5).to(torch.uint8).permute(1,2,0).numpy()).save(path/f'{i:03d}.png')
    for kind,data in [('tensor',directory/'images.pt'),('folder',folder)]:
        config=ROOT/'configs/controlled/ordinary_flow.json';full=directory/kind/'full';split=directory/kind/'resumed'
        common=['--config',str(config),'--data',str(data),'--batch-size','8','--seed','42']
        for count,out,resume in [(4,full,False),(2,split,False),(2,split,True)]:
            command(['-m','umdiff.cli','train',*common,'--steps',str(count),'--output',str(out),*(['--resume',str(split/'checkpoint.pt')] if resume else [])])
        a=torch.load(full/'checkpoint.pt',weights_only=False);b=torch.load(split/'checkpoint.pt',weights_only=False)
        if not equal(a,b):raise AssertionError(f'Local {kind} checkpoint resume differs')
    data={'latents':torch.randn(19,4,16,16,generator=generator),'conditions':{'text':torch.randn(19,5,8,generator=generator),'mask':torch.ones(19,5)}}
    torch.save(data,directory/'latents.pt');config=json.loads((ROOT/'configs/native_fixtures/mdm.json').read_text())
    config['training']['conditioning']['conditioning']={'kind':'field','field':'text'};config['training']['conditioning']['cond_mask']={'kind':'field','field':'mask'}
    config_path=directory/'mdm_fields.json';config_path.write_text(json.dumps(config,indent=2));out=directory/'mdm_fields'
    command(['-m','umdiff.cli','train','--config',str(config_path),'--data',str(directory/'latents.pt'),'--steps','4','--output',str(out)])
    torch.save({k:v[:2] for k,v in data['conditions'].items()},directory/'sample_conditions.pt')
    command(['-m','umdiff.cli','sample','--checkpoint',str(out/'checkpoint.pt'),'--conditioning',str(directory/'sample_conditions.pt'),'--output',str(out/'sample.pt')])
    for name in ('speed','mrflow','nami'):
        command([str(Path(__file__).resolve()),'--workflow',name])
        asset_dir=directory/name
        command(['-m','umdiff.cli','infer','--config',str(asset_dir/'config.json'),'--embeddings',str(asset_dir/'embeddings.pt'),'--output',str(asset_dir/'cli_output.pt'),'--seed','42'])
    command([str(ROOT/'tools/validate_rmd_local_assets.py')])
    print(json.dumps({'status':'passed','cases':len(cases),'device':'cpu','gpu_used':False}))
if __name__=='__main__':main()
