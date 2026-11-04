#!/usr/bin/env python3
"""Check real architecture schemas on meta device; no weights/data downloads."""
import argparse,json,sys,subprocess,os,traceback
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--recipe',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.recipe:
        import torch
        from umdiff import build
        from umdiff.cli import read_config,make_method
        c=read_config(args.recipe);model=deepcopy(c['model']);local=model['name'].startswith('local_');initialization_deferred=[]
        if model['name']=='native_mdm':
            inner=model.get('args',{}).get('config',{})
            if inner.get('initialize_inner_with_pretrained'):
                initialization_deferred.append(inner['initialize_inner_with_pretrained'])
                inner['initialize_inner_with_pretrained']=None
        try:
            if c.get('task') not in ('codec','rmd'):make_method(c)
            if not local:
                with torch.device('meta'):network=build('model',model)
                parameters=sum(p.numel() for p in network.parameters())
            else:parameters=None
            print(json.dumps({'status':'passed','parameters':parameters,'local_asset_model_deferred':local,'initialization_deferred':initialization_deferred,'runtime_assets_verified':False}));return 0
        except Exception:print(traceback.format_exc());return 1
    results=[];os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
    for path in sorted((ROOT/'configs/paper_recipes').glob('*.json')):
        if path.name=='INDEX.json':continue
        result=subprocess.run([sys.executable,__file__,'--recipe',str(path)],capture_output=True,text=True)
        results.append({'recipe':path.stem,'status':'passed' if result.returncode==0 else 'failed','output':result.stdout,'stderr':result.stderr[-500:]})
        print(json.dumps({'recipe':path.stem,'status':results[-1]['status']}),flush=True)
    out=args.output or ROOT/'evidence/recipe_extensions/recipe_schema.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps({'status':'passed' if all(r['status']=='passed' for r in results) else 'failed','recipes':results},indent=2));return int(any(r['status']!='passed' for r in results))
if __name__=='__main__':raise SystemExit(main())
