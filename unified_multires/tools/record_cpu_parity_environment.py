#!/usr/bin/env python3
"""Read-only author snapshot verification and CPU environment manifest."""
import argparse,hashlib,importlib.metadata,json,os,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES']=''
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    pins=json.loads((ROOT/'evidence/sources.json').read_text());rows=[]
    for name,repo in pins.items():
        selected=repo.get('selected_files',{})
        mismatches=[f for f,h in selected.items() if sha(ROOT/'sources'/name/f)!=h]
        assert not mismatches,(name,mismatches)
        rows.append(dict(repo=name,commit=repo.get('commit'),verified=len(selected),mismatches=mismatches))
    (a.output/'source_integrity.json').write_text(json.dumps(dict(status='passed',repositories=rows,selected_files=sum(x['verified'] for x in rows),manifest_sha256=sha(ROOT/'evidence/sources.json')),indent=2)+'\n')
    import torch
    versions={}
    for name in ('torch','torchvision','numpy','diffusers','transformers','accelerate','timm','flash-attn','torchdiffeq','torchdata','blobfile','lxml','pycryptodomex'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    assert not torch.cuda.is_initialized()
    (a.output/'environment.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,host=platform.node(),platform=platform.platform(),versions=versions,device='cpu',CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cuda_initialized=False,torch_threads=torch.get_num_threads(),shared_environment_modified=False),indent=2)+'\n')
    print(json.dumps(dict(status='passed',repositories=len(rows),source_files=sum(x['verified'] for x in rows),cuda_initialized=False)))
if __name__=='__main__':main()
