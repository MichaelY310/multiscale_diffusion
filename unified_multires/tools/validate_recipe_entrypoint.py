#!/usr/bin/env python3
"""Local shard traversal/resume and recipe binding; synthetic local fixture."""
import os,json,subprocess,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import torch
from umdiff.cli import read_config,train
from validate_cli import equal

def main():
    torch.set_num_threads(1);run=ROOT/'runs/recipe_extensions/entrypoint';run.mkdir(parents=True,exist_ok=True)
    c=read_config(ROOT/'configs/controlled/ordinary_flow.json');c['image_shape']=[3,8,8];c.setdefault('training',{})['data']={'shuffle':True}
    cfg=run/'config.json';cfg.write_text(json.dumps(c));data=ROOT/'runs/recipe_extensions/auxiliary/shards'
    def args(out,steps,resume=None):return SimpleNamespace(config=str(cfg),data=str(data),synthetic=False,output=str(out),resume=resume,steps=steps,batch_size=4,seed=88,device='cpu',log_every=10,save_every=0)
    train(args(run/'full',4));train(args(run/'split',2));train(args(run/'split',2,str(run/'split/checkpoint.pt')))
    a=torch.load(run/'full/checkpoint.pt',weights_only=False);b=torch.load(run/'split/checkpoint.pt',weights_only=False)
    assert all(equal(a[k],b[k]) for k in ('model','ema','optimizer','stream','torch_rng','numpy_rng','python_rng','data_binding','images_seen'))
    result=subprocess.run([sys.executable,str(ROOT/'tools/render_paper_recipe.py'),'dctdiff_cifar10_uvit_small_2by2','--data',str(data),'--output',str(run/'bound_dct.json')],capture_output=True,text=True);assert result.returncode==0,result.stderr
    blocked=subprocess.run([sys.executable,'-m','umdiff.cli','train','--config',str(run/'bound_dct.json'),'--synthetic','--steps','1','--output',str(run/'forbidden')],capture_output=True,text=True);assert blocked.returncode!=0 and 'require local real data' in blocked.stderr
    report={'status':'passed','device':'cpu','gpu_used':False,'synthetic_local_shard_data':True,'shuffle_resume_all_states_exact':True,'recipe_renderer_source_hash_check':'passed','formal_synthetic_training_rejected':True,'quality_verified':False}
    (ROOT/'evidence/recipe_extensions/entrypoint.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
