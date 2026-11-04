#!/usr/bin/env python3
"""Bind existing local assets, verify pinned recipe sources, emit runnable JSON."""
import argparse,json,re,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
def main():
    p=argparse.ArgumentParser();p.add_argument('recipe');p.add_argument('--asset',action='append',default=[],metavar='NAME=PATH');p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    path=Path(a.recipe)
    if not path.is_file():path=ROOT/'configs/paper_recipes'/(a.recipe+'.json')
    raw=path.read_text();config=json.loads(raw);required=set(re.findall(r'@([a-z][a-z0-9_]*)@',raw));assets={}
    for item in a.asset:
        key,value=item.split('=',1);asset=Path(value).resolve()
        if not asset.exists():raise FileNotFoundError(asset)
        assets[key]=str(asset)
    if set(assets)!=required:raise ValueError(f'Required assets {sorted(required)}; provided {sorted(assets)}')
    if not a.data.exists():raise FileNotFoundError(a.data)
    for source in config['evidence'].get('source_pins',[]):
        root=ROOT.parent if source.get('root')=='workspace' else ROOT/'sources'
        if hashlib.sha256((root/(source['file'].removeprefix('multiscale_diffusion/') if source.get('root')=='workspace' else source['file'])).read_bytes()).hexdigest()!=source['sha256']:raise ValueError('Source snapshot changed: '+source['file'])
    def replace(value):
        if isinstance(value,str):return re.sub(r'@([a-z][a-z0-9_]*)@',lambda match:assets[match[1]],value)
        if isinstance(value,list):return [replace(v) for v in value]
        if isinstance(value,dict):return {k:replace(v) for k,v in value.items()}
        return value
    config=replace(config);config['evidence']['local_data']=str(a.data.resolve());config['evidence']['local_assets']=assets
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(config,indent=2)+'\n');print(json.dumps({'config':str(a.output),'data':str(a.data),'quality_verified':False}))
if __name__=='__main__':main()
