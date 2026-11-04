#!/usr/bin/env python3
"""Actual spawn workers, native PyTorch reader parity, fresh-process resume."""
import argparse,hashlib,json,os,random,subprocess,sys,tempfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff.data_pipeline import StatefulOfflineLoader


class StochasticFiles(torch.utils.data.Dataset):
    def __init__(self,directory):self.files=sorted(Path(directory).glob('*.npy'))
    def __len__(self):return len(self.files)
    def __getitem__(self,index):
        value=torch.from_numpy(np.load(self.files[index]))
        return dict(id=index,image=value,python=random.random(),numpy=np.random.random(),torch=torch.rand(3))


def bytes_hash(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str((x.shape,x.dtype)).encode());h.update(x.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x,dict):
            for k,v in x.items():h.update(str(k).encode());visit(v)
        else:h.update(repr(x).encode())
    visit(value);return h.hexdigest()


def rng():return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate())
def restore(value):torch.set_rng_state(value['torch']);np.random.set_state(value['numpy']);random.setstate(value['python'])


def worker(a):
    torch.set_num_threads(1);torch.manual_seed(871);np.random.seed(871);random.seed(871)
    c=json.loads(a.contract.read_text());dataset=StochasticFiles(a.data);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    if a.native:
        sampler=torch.utils.data.DistributedSampler(dataset,num_replicas=c['world_size'],rank=c['rank'],shuffle=c['shuffle'],seed=c['seed'],drop_last=False)
        kw=dict(batch_size=c['batch_size'],sampler=sampler,drop_last=c['drop_last'],num_workers=c['num_workers'],generator=torch.Generator().manual_seed(c['seed']))
        if c['num_workers']:kw.update(multiprocessing_context='spawn',prefetch_factor=2)
        loader=torch.utils.data.DataLoader(dataset,**kw)
        epoch=0;sampler.set_epoch(epoch);iterator=iter(loader)
        def next_value():
            nonlocal iterator,epoch
            try:return next(iterator)
            except StopIteration:
                epoch+=1;sampler.set_epoch(epoch);iterator=iter(loader);return next(iterator)
    else:
        loader=StatefulOfflineLoader(dataset,**c)
        if a.resume:
            value=torch.load(a.resume,weights_only=False);loader.load_state_dict(value['reader']);restore(value['rng'])
        next_value=lambda:next(loader)
    rows=[]
    for i in range(a.steps):
        batch=next_value();rows.append(dict(batch_sha256=bytes_hash(batch),ids=batch['id'].tolist()))
    (out/'batches.json').write_text(json.dumps(rows,indent=2)+'\n')
    if not a.native:torch.save(dict(reader=loader.state_dict(),rng=rng()),out/'reader.pt')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--worker',action='store_true')
    p.add_argument('--contract',type=Path);p.add_argument('--data',type=Path);p.add_argument('--steps',type=int,default=10)
    p.add_argument('--resume',type=Path);p.add_argument('--native',action='store_true');a=p.parse_args()
    if a.worker:worker(a);return 0
    a.output.mkdir(parents=True,exist_ok=True);started=time.time();rows=[]
    with tempfile.TemporaryDirectory(prefix='umdiff_worker_data_') as tmp:
        temp=Path(tmp);data=temp/'files';data.mkdir()
        for i in range(11):np.save(data/f'{i:02}.npy',np.arange(45,dtype=np.float32).reshape(3,3,5)+i)
        cases=[(w,r,world,drop,shuffle) for w in (0,2) for r,world in ((0,1),(0,2),(1,2)) for drop,shuffle in ((True,True),(False,False))]
        for i,(w,r,world,drop,shuffle) in enumerate(cases):
            name=f'w{w}_rank{r}of{world}_drop{drop}_shuffle{shuffle}';base=temp/name;base.mkdir()
            contract=dict(batch_size=2,seed=912,shuffle=shuffle,drop_last=drop,num_workers=w,rank=r,world_size=world)
            cp=base/'contract.json';cp.write_text(json.dumps(contract))
            def run(kind,steps,resume=None,native=False):
                cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--contract',str(cp),'--data',str(data),'--output',str(base/kind),'--steps',str(steps)]
                if resume:cmd+=['--resume',str(resume)]
                if native:cmd+=['--native']
                with (a.output/(name+'_'+kind+'.log')).open('w') as log:
                    result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':''},timeout=180)
                if result.returncode:raise RuntimeError(f'{name}/{kind} failed; inspect log')
                return json.loads((base/kind/'batches.json').read_text())
            native_rows=run('native',10,native=True);full=run('full',10);first=run('first',3);resumed=run('resumed',7,base/'first/reader.pt')
            if native_rows!=full:raise AssertionError(f'Native DataLoader differs: {name}')
            if full!=first+resumed:raise AssertionError(f'Fresh process reader differs: {name}')
            wrong=StatefulOfflineLoader(StochasticFiles(data),**{**contract,'seed':913})
            try:wrong.load_state_dict(torch.load(base/'first/reader.pt',weights_only=False)['reader'])
            except ValueError:guard=True
            else:raise AssertionError('Reader contract mismatch accepted')
            row=dict(case=name,status='passed',contract=contract,native_reader_byte_exact=True,
                fresh_process_resume_byte_exact=True,wrong_contract_rejected=guard,steps=10,batches=full)
            rows.append(row);print(json.dumps({k:row[k] for k in ('case','status')}),flush=True)
            (a.output/'SUMMARY.json').write_text(json.dumps(dict(status='running',cases=rows),indent=2)+'\n')
    report=dict(status='passed',gpu_used=False,formal_training=False,cases=rows,seconds=time.time()-started,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Native torch DataLoader + DistributedSampler,0/2 actual spawn workers,rank0/1,shuffle/drop_last,local generated files and three RNGs; fresh processes. Does not certify private streaming readers or CUDA/NCCL.')
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='passed',cases=len(rows),seconds=report['seconds'])),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
