#!/usr/bin/env python3
"""Read bounded ZIP/pickle metadata ranges, never model tensor payloads."""
import argparse
import collections
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
import urllib.request
import zipfile

ROOT=Path(__file__).resolve().parents[1]
PLAN=json.loads((ROOT/'docs/experiment_registry/hf_rename_plan.json').read_text())
MODELS=json.loads((ROOT/'docs/experiment_registry/models.json').read_text())
GROUPS={m['legacy_hf_group']:m for m in MODELS}
GROUPS['RDD-Multistage-100-300-600-8x8-continued-from400k']=GROUPS['RDD-Multistage-100-300-600']
OUT=ROOT/'docs/validation/20261104/CHECKPOINT_NAMING_METADATA.json'

class Remote(io.RawIOBase):
    def __init__(self,url,size):self.url=url;self.size=size;self.pos=0;self.total=0
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,n,whence=0):
        self.pos=n if whence==0 else self.pos+n if whence==1 else self.size+n
        return self.pos
    def read(self,n=-1):
        n=min(n if n>=0 else self.size-self.pos,self.size-self.pos)
        if n==0:return b''
        if not (0<n<4*1024**2 and self.total+n<8*1024**2):raise ValueError('Metadata read limit exceeded')
        start,end=self.pos,self.pos+n-1
        req=urllib.request.Request(self.url+f'?naming_metadata={start}_{end}',headers={'Range':f'bytes={start}-{end}'})
        with urllib.request.urlopen(req,timeout=40) as response:
            if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {start}-{end}/{self.size}':raise ValueError('Server did not honor bounded range')
            data=response.read(n+1)
        if len(data)!=n:raise ValueError('Unexpected range size')
        self.pos+=n;self.total+=n;return data

class Safe(pickle.Unpickler):
    def persistent_load(self,p):
        if p[0]!='storage':raise ValueError('Unexpected persistent object')
        return {'dtype':p[1],'key':p[2],'device':p[3],'numel':p[4]}
    def find_class(self,module,name):
        if (module,name)==('collections','OrderedDict'):return collections.OrderedDict
        if (module,name)==('argparse','Namespace'):return argparse.Namespace
        if module=='torch' and name.endswith('Storage'):return name
        if module=='torch._utils' and name in ('_rebuild_tensor','_rebuild_tensor_v2','_rebuild_tensor_v3'):
            return lambda storage,offset,size,stride,*args:{'shape':list(size)}
        raise ValueError('Blocked pickle global '+module+'.'+name)

def values(v):return [int(x) for x in re.split('[,/]',v)] if isinstance(v,str) else list(v)

def check(item):
    path=item['old_path'];m=GROUPS[path.split('/')[1]]
    result={'old_path':path,'canonical_path':item['new_path'],'sha256':item['sha256'],'status':'unverified'}
    url=f"https://huggingface.co/{PLAN['repo_id']}/resolve/{PLAN['source_revision']}/{path}"
    remote=Remote(url,item['size'])
    try:
        with zipfile.ZipFile(remote) as z:
            info=next(i for i in z.infolist() if i.filename.endswith('/data.pkl'))
            if info.file_size>4*1024**2:raise ValueError('Pickle metadata exceeds cap')
            raw=z.read(info)
        x=Safe(io.BytesIO(raw)).load()
        args=x.get('args',{})
        if isinstance(args,argparse.Namespace):args=vars(args)
        if not isinstance(args,dict):args={}
        fields=('stage_lengths','resolutions','s_boundaries','bounds','resolution','model','diffusion_steps','T_c','T_c_plus_1','stage_idx','train_steps','step','global_step')
        metadata={k:x[k] for k in fields if k in x and not isinstance(x[k],dict)};metadata['args']={k:args[k] for k in fields if k in args}
        result.update(metadata=metadata,metadata_sha256=hashlib.sha256(raw).hexdigest())
        state=x.get('model',{})
        if isinstance(state,dict):
            embeds=[v.get('shape') for k,v in state.items() if k.endswith('x_embedder.proj.weight')]
            blocks={int(b.group(1)) for k in state for b in [re.search(r'(?:^|\.)blocks\.(\d+)\.',k)] if b}
            result['model_structure']={'embedding_weight_shape':embeds,'transformer_depth':max(blocks)+1 if blocks else None,'parameter_entries':len(state)}
        lengths=x.get('stage_lengths',args.get('stage_lengths'));bounds=x.get('s_boundaries',args.get('s_boundaries',args.get('bounds')))
        checks=[]
        if lengths is not None:checks.append(values(lengths)==m['native_stage_lengths'])
        if bounds is not None and m['family']=='fm':
            expected=[m['training_stages'][0]['start']]+[s['end'] for s in m['training_stages']]
            checks.append(len(bounds)==len(expected) and all(abs(float(a)-b)<1e-8 for a,b in zip(bounds,expected)))
        result['status']='matches_saved_clock' if checks and all(checks) else 'clock_mismatch' if checks else 'clock_not_saved'
    except Exception as e:result['error']=type(e).__name__+': '+str(e)
    result['bytes_read_metadata_only']=remote.total
    return result

def main():
    items=[r for r in PLAN['operations'] if r['kind']=='checkpoint' and r['old_path'].endswith('.pt')]
    previous={r['old_path']:r for r in json.loads(OUT.read_text())['checkpoints']} if OUT.exists() else {}
    results=[previous[r['old_path']] for r in items if r['old_path'] in previous and previous[r['old_path']]['status']!='unverified']
    todo=[r for r in items if r['old_path'] not in {x['old_path'] for x in results}]
    def save():
        for r in results:
            if 'model_structure' not in r:continue
            group=r['old_path'].split('/')[1]
            if 'Multistage' in group:
                match=re.search(r'checkpoints_(\d+)x',r['old_path'])
                if not match:continue
                expected={32:(1152,28),16:(1024,24),8:(768,12)}[int(match.group(1))]
            else:expected=(1152,28) if 'XL' in group else (1024,24)
            shape=r['model_structure'];r['architecture_matches_canonical_name']=(shape['embedding_weight_shape']==[[expected[0],4,2,2]] and shape['transformer_depth']==expected[1])
        counts=dict(collections.Counter(r['status'] for r in results))
        OUT.write_text(json.dumps({'scope':'Bounded metadata-only archive reads. No tensor data or inference. Missing metadata is not silently treated as a verified clock.','counts':counts,'bytes_read_metadata_only':sum(r['bytes_read_metadata_only'] for r in results),'checkpoints':sorted(results,key=lambda r:r['old_path'])},indent=2,default=str)+'\n')
        return counts
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(check,r) for r in todo]
        for f in as_completed(fs):
            results.append(f.result())
            if len(results)%20==0:print(json.dumps({'completed':len(results),'counts':save()}),flush=True)
    print(json.dumps({'completed':len(results),'counts':save()}),flush=True)

if __name__=='__main__':main()
