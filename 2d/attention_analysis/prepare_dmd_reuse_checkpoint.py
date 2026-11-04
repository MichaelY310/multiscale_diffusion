"""Extract a stable generator snapshot; original latest.pt is read-only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();src=Path(a.checkpoint);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    before=src.stat();ck=torch.load(src,map_location='cpu',weights_only=False,mmap=True)
    assert ck['distillation']=='RDD-Original-DMD-3NFE-FullOutput-v2'
    assert tuple(ck['stage_lengths'])==(300,300,400) and tuple(ck['resolutions'])==(32,16,8)
    state=ck['student']; keys=list(state)
    for k in keys: torch.testing.assert_close(state[k],ck['ema'][k],atol=0,rtol=0)
    audit={'source':str(src),'source_bytes':before.st_size,'checkpoint_step':ck['step'],
           'distillation':ck['distillation'],'stage_lengths':ck['stage_lengths'],'resolutions':ck['resolutions'],
           'generator_source':'student','ema_equals_student':True,'original_args':ck['args'],
           'parameters':sum(v.numel() for v in state.values()),'keys':keys,
           'shape_examples':{k:list(state[k].shape) for k in ('x_embedder.proj.weight','blocks.0.attn.qkv.weight','final_layer.linear.weight')}}
    destination=out/'generator_start.pt'
    if not destination.exists():
        tmp=out/f'generator_start.tmp-{os.getpid()}'
        torch.save({'model':state,'step':ck['step'],'stage_lengths':ck['stage_lengths'],'resolutions':ck['resolutions'],'source':str(src)},tmp)
        os.replace(tmp,destination)
    else:
        existing=torch.load(destination,map_location='cpu',weights_only=True,mmap=True)
        assert int(existing['step'])==int(ck['step']),'Refuse to relabel an old snapshot'
        for k in state:torch.testing.assert_close(existing['model'][k],state[k],atol=0,rtol=0)
        del existing
    h=hashlib.sha256()
    with destination.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    after=src.stat();assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns),'Original checkpoint changed during inspection'
    audit['snapshot_sha256']=h.hexdigest()
    (out/'checkpoint_audit.json').write_text(json.dumps(audit,indent=2,default=str))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('keys','original_args')},indent=2),flush=True)
