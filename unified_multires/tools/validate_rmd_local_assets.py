#!/usr/bin/env python3
"""Three independent local RMD weights, real embedding loader, resume and sample."""
import json,os,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from validate_cli import equal
from umdiff.registry import build
from umdiff.distillation import RMDTrainer

def main():
    torch.set_num_threads(2);directory=ROOT/'runs/offline_entrypoints/rmd';directory.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'configs/native_fixtures/rmd.json').read_text());weights={};expected={}
    for i,key in enumerate(('model','fake','real')):
        torch.manual_seed(100+i);model=build('model',config['model'])
        expected[key]={k:v.clone() for k,v in model.state_dict().items()}
        path=directory/f'{key}.pt';torch.save(expected[key],path);weights[key]=str(path)
    config['checkpoints']=weights;config_path=directory/'config.json';config_path.write_text(json.dumps(config,indent=2))
    g=torch.Generator().manual_seed(17)
    data={'prompt_embeds':torch.randn(7,3,8,generator=g),'negative_prompt_embeds':torch.randn(1,3,8,generator=g)}
    torch.save(data,directory/'embeddings.pt');torch.save({'prompt_embeds':data['prompt_embeds'][:2]},directory/'sample.pt')
    trainer=RMDTrainer(config,build('model',config['model']),data['negative_prompt_embeds'])
    for key,state in expected.items():
        if not equal(state,getattr(trainer,key).state_dict()):raise AssertionError(f'Wrong local initialization for {key}')
    env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(ROOT),str(ROOT/'_deps')]))
    records=[]
    def command(argv):
        result=subprocess.run([sys.executable,'-m','umdiff.cli',*argv],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
        records.append(dict(argv=argv,exit_code=result.returncode,log=result.stdout[-3000:]))
        if result.returncode:raise RuntimeError(result.stdout)
    full=directory/'full';split=directory/'resumed'
    common=['--config',str(config_path),'--data',str(directory/'embeddings.pt'),'--batch-size','2','--seed','42']
    for steps,out,resume in [(4,full,False),(2,split,False),(2,split,True)]:
        command(['train',*common,'--steps',str(steps),'--output',str(out),*(['--resume',str(split/'checkpoint.pt')] if resume else [])])
    a=torch.load(full/'checkpoint.pt',weights_only=False);b=torch.load(split/'checkpoint.pt',weights_only=False)
    if not equal(a,b):raise AssertionError('Local RMD resume state differs')
    if not equal(a['real'],expected['real']):raise AssertionError('Frozen teacher changed')
    for key in ('model','fake'):
        if equal(a[key],expected[key]):raise AssertionError(f'{key} did not update')
    for out in (full,split):command(['sample','--checkpoint',str(out/'checkpoint.pt'),'--conditioning',str(directory/'sample.pt'),'--count','2','--output',str(out/'sample.pt')])
    if not equal(torch.load(full/'sample.pt',weights_only=True),torch.load(split/'sample.pt',weights_only=True)):raise AssertionError('Local prompt sampling differs after resume')
    result=dict(status='passed',device='cpu',gpu_used=False,three_local_models_loaded_bitwise=True,frozen_teacher_unchanged=True,both_trainable_models_updated=True,full_checkpoint_resume_bitwise=True,conditional_sample_resume_bitwise=True,commands=records)
    (ROOT/'evidence/rmd_offline_assets.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='commands'}))
if __name__=='__main__':main()
