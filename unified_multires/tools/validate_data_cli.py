#!/usr/bin/env python3
"""Fresh-process training CLI resume through each offline data reader.

Generates tiny fixtures; removes tensor checkpoints/pictures after hashing.
"""
import argparse,hashlib,json,os,subprocess,sys,tempfile,time,traceback,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from PIL import Image
from validate_fresh_process_lifecycle import manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    started=time.time();rows=[]
    for kind in ('tensor','shards','images','moments','moments_cfg','relay_zip'):
        for workers in (0,2):
            name=f'{kind}_workers{workers}';case=a.output/name;case.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(prefix='umdiff_data_cli_') as tmp:
                tmp=Path(tmp);generator=torch.Generator().manual_seed(871)
                c=json.loads((ROOT/'configs/controlled/ordinary_ddpm.json').read_text());c['image_shape']=[3,16,16];c['model']['args']['width']=8
                c['training']['optimizer'].update(accumulation_steps=2,randomness='native_global')
                c['training']['data']=dict(reader='stateful_author',num_workers=workers,shuffle=True,drop_last=True,
                    image_size=16,crop='adm_center',horizontal_flip=True,flip_rng='torch',normalization='torchvision')
                images=torch.randn(11,3,16,16,generator=generator);labels=torch.arange(11)%2
                if kind=='tensor':data=tmp/'data.pt';torch.save({'images':images,'labels':labels},data)
                else:
                    data=tmp/'data';data.mkdir()
                    if kind=='shards':
                        entries=[]
                        for i,(lo,hi) in enumerate(((0,3),(3,8),(8,11))):
                            torch.save({'images':images[lo:hi],'labels':labels[lo:hi]},data/f'{i}.pt');entries.append(dict(file=f'{i}.pt',count=hi-lo))
                        (data/'manifest.json').write_text(json.dumps(dict(format='umdiff_shards_v1',shards=entries)))
                    elif kind in ('images','relay_zip'):
                        rng=np.random.default_rng(71)
                        for i in range(11):
                            directory=data/str(i%2);directory.mkdir(exist_ok=True)
                            Image.fromarray(rng.integers(0,256,(27,39,3),dtype=np.uint8)).save(directory/f'{i}.png')
                        if kind=='relay_zip':
                            # The author stores class identities in dataset.json,
                            # independently of the folder names.
                            pairs=[]
                            for i,path in enumerate(sorted(data.rglob('*.png'))):
                                Image.open(path).resize((16,16)).save(path);pairs.append([str(path.relative_to(data)),i%2])
                            (data/'dataset.json').write_text(json.dumps({'labels':pairs}));archive=tmp/'relay.zip'
                            with zipfile.ZipFile(archive,'w') as z:
                                for path in sorted(data.rglob('*')):
                                    if path.is_file():z.write(path,path.relative_to(data))
                            data=archive;c['training']['data'].update(kind='relay_zip',author_args=dict(resolution=16,use_labels=True,xflip=True))
                    else:
                        c['training']['data']['kind']='moments_npy';c['training']['input_transform']={'name':'vae_moments','args':{'scale':.83}}
                        c['training']['input_transform_scope']='microbatch'
                        if kind=='moments_cfg':c['training']['data']['cfg_dropout']=.1
                        for i in range(11):
                            value=np.empty(2,dtype=object);value[:]=[torch.cat([images[i],torch.zeros_like(images[i])]).numpy(),int(labels[i])];np.save(data/f'{i}.npy',value)
                config=tmp/'config.json';config.write_text(json.dumps(c));commands=[]
                def run(mode,steps,resume=None):
                    out=tmp/mode;cmd=[sys.executable,'-m','umdiff.cli','train','--config',str(config),'--data',str(data),'--batch-size','4','--device','cpu','--seed','17','--steps',str(steps),'--output',str(out),'--save-every','1']
                    if resume:cmd+=['--resume',str(resume)]
                    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
                    with (case/(mode+'.log')).open('w') as log:process=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
                    commands.append(dict(argv=cmd,returncode=process.returncode))
                    if process.returncode:raise RuntimeError(name+'/'+mode+' failed')
                    return torch.load(out/'checkpoint.pt',weights_only=False),out/'checkpoint.pt'
                try:
                    full,full_path=run('full',4);first,first_path=run('first',2);resumed,resumed_path=run('resumed',2,first_path)
                    if manifest(full)!=manifest(resumed):raise AssertionError('Complete checkpoint state differs across fresh-process resume')
                    rows.append(dict(name=name,status='passed',fresh_process_exact=True,all_checkpoint_fields_exact=True,
                        crossed_shard_boundary=kind=='shards',crossed_epoch_boundary=kind!='relay_zip',reader_contract=full['data_binding']['reader_contract'],
                        checkpoint_bytes=full_path.stat().st_size,checkpoint_sha256=hashlib.sha256(full_path.read_bytes()).hexdigest(),
                        state_manifest=manifest(full),effective_config=c,commands=commands))
                except Exception:rows.append(dict(name=name,status='failed',failure=traceback.format_exc(),commands=commands))
                print(json.dumps({k:v for k,v in rows[-1].items() if k in ('name','status','failure')}),flush=True)
                (a.output/'SUMMARY.json').write_text(json.dumps(dict(status='running',cases=rows),indent=2)+'\n')
    report=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',cases=rows,gpu_used=False,formal_training=False,
        seconds=time.time()-started,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),temporary_payloads_deleted=True)
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
