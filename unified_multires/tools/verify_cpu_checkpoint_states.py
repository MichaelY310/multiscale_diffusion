#!/usr/bin/env python3
"""Read back paired temporary checkpoints; strict unfiltered state comparison."""
import argparse,hashlib,json,os,sys
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
from activation_trace import byte_view
from cleanup_cpu_parity_payloads import refs,sha

def compare(a,b,path,stats):
    assert type(a) is type(b),(path,type(a),type(b))
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype,(path,a.shape,b.shape,a.dtype,b.dtype)
        assert torch.equal(byte_view(a),byte_view(b)),path
        stats['tensors']+=1;stats['tensor_bytes']+=a.numel()*a.element_size()
        stats[path.split('/')[0]+'_tensors']+=1
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],path+'/'+str(k) if path else str(k),stats)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'/'+str(i),stats)
    else:
        assert a==b,(path,a,b)
        stats['scalars']+=1

def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();base=a.evidence.resolve();torch.set_num_threads(2)
    found=set()
    for sub in ('long_release','extensions_current','codec_current'):
        for f in (base/sub).rglob('*.json'):
            obj=json.loads(f.read_text())
            if obj.get('status')=='passed':found.update(refs(obj))
    rows=[]
    for rel in sorted(found):
        directory=(ROOT/rel).resolve();assert directory.is_relative_to(ROOT/'runs')
        native=directory/'native.pt';unified=directory/'unified.pt'
        aa=torch.load(native,map_location='cpu',weights_only=True);bb=torch.load(unified,map_location='cpu',weights_only=True);stats=Counter();compare(aa,bb,'',stats)
        rows.append(dict(checkpoint_pair=rel,status='passed',native_sha256=sha(native),unified_sha256=sha(unified),native_bytes=native.stat().st_size,unified_bytes=unified.stat().st_size,checks=dict(stats)))
        print(json.dumps(dict(checkpoint_pair=rel,status='passed',tensors=stats['tensors'])),flush=True)
        del aa,bb
    assert rows
    report=dict(status='passed',pairs=len(rows),rows=rows,device='cpu',gpu_used=False,cuda_initialized=torch.cuda.is_initialized(),scope='Raw model/EMA/Adam state bytes and exact typed scalars/container structure, including all stored optimizer param_groups; no named/frozen group filtering. Native/unified controlled CPU checkpoints, not full official CLI resume or pretrained checkpoints. ZIP serialization SHA reported separately, not asserted equal.')
    assert not report['cuda_initialized']
    (base/'checkpoint_state_bytes.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',pairs=len(rows))))
if __name__=='__main__':main()
