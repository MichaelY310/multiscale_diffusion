"""Fixed reader vs original RDD on new, paired10K trajectories; quality only.

No memory-refiner, no extra training, no inference timing claims. This separately
named protocol never alters existing sampling scripts or checkpoint contracts.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

from sample_csra_prefix_fid_pilot import sha, write_json

ROOT = Path('eval_results/csra_reader_independent_fid10k_20260914')
VERSION = 'csra_front1_reader_independent_fid10k_fixed400_v1'
ARMS = ('original_rdd', 'reader')
BASE_SHA = 'cade56f41317b2d04045431c139b2dd0097346e6392142a39f189274187a0c9f'
READER = Path('checkpoints/RDD-CSRA-state-replay-teacher_only-L400K-20260914/latest.pt')
READER_SHA = '87b602ecb9bebae131a7117eff9b6077b8ffca06cb4b6d81654cf266557818b5'
SEED_BASE = 310609140


def batches(rank, smoke=False):
    if rank not in range(4): raise ValueError('Only four shards')
    count = 4 if smoke else 2500
    return [(i, min(64, count-64*i), SEED_BASE + int(smoke)*10000000 + rank*100000 + i*1000)
            for i in range((count+63)//64)]


def protocol(smoke):
    return dict(version=VERSION, smoke=smoke, arms=list(ARMS), count=16 if smoke else 10000,
        base_sha256=BASE_SHA, reader_sha256=READER_SHA, reader_additional_updates=400,
        original_checkpoint_step=400000, model='RDD Unified-L/2', train_time_ranges='300/300/400 fine/middle/coarse',
        executed_fine_blocks=[0]+list(range(7,24)), skipped_fine_blocks=list(range(1,7)),
        source='original16,t300,fullhidden block13', neighbor=False, memory_refiner=False,
        steps=[80,80,90], sampler='DDPM', cfg=1., boundary='noise_filling', tf32=False,
        seed_base=SEED_BASE, batch_size=64, tail_batch=4, vae='sd-vae-ft-ema/BF16/scale0.18215', decode_batch=4)


def check_meta(meta, dest, index, count, seed, provenance):
    if (not meta.get('complete') or meta.get('index')!=index or meta.get('count')!=count
        or meta.get('seed')!=seed or meta.get('provenance')!=provenance
        or len(meta.get('labels',[]))!=count or meta.get('sha256')!=sha(dest)):
        raise ValueError(f'Corrupt or changed completed chunk:{dest}')


def sample(args):
    import numpy as np
    import torch
    from sample_csra_prefix_quality_pilot import LabelledRollout, BASE
    from sample_csra_adapter_preview import sample_rdd, decode, to_pil, AutoencoderKL
    from train_csra_state_replay import load_result
    from train_rdd_csra_adapter import load_frozen_teacher
    if os.environ.get('CUDA_VISIBLE_DEVICES') != str(args.rank):
        raise ValueError('One worker per matching physicalGPU0-3')
    if sha(BASE)!=BASE_SHA or sha(READER)!=READER_SHA: raise ValueError('Fixed checkpoint changed')
    if shutil.disk_usage('.').free < 16*2**30: raise RuntimeError('Need16GiB free disk; no deletion')
    torch.set_num_threads(2)
    dev=torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 20*2**30: raise RuntimeError('Need20GiB GPU headroom')
    torch.cuda.set_per_process_memory_fraction(.18,dev)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=args.root/f'shard_{args.rank}';out.mkdir(parents=True,exist_ok=True)
    plans=batches(args.rank,args.smoke); seeds={s for _,_,s in plans}
    for root in ('eval_results/csra_rollout_data4k_20260912','eval_results/csra_state_replay_20260914/data'):
        for m in Path(root).glob('shard_*/manifest.json'):
            if any(row['seed'] in seeds for row in json.loads(m.read_text())['files']):
                raise ValueError('Seed overlap with training or validation corpus')
    provenance=dict(protocol=protocol(args.smoke), sampler_sha256=sha(Path(__file__)),
        shared_sampler_sha256=sha('2d/attention_analysis/sample_csra_adapter_preview.py'),
        labelled_wrapper_sha256=sha('2d/attention_analysis/sample_csra_prefix_quality_pilot.py'))
    config=out/'config.json'
    if config.exists() and json.loads(config.read_text())!=provenance: raise ValueError('Cannot mix run versions')
    write_json(config,provenance)
    if (out/'complete.json').exists():
        if json.loads((out/'complete.json').read_text()) != dict(complete=True,**provenance):
            raise ValueError('Wrong completion marker')
        # Still audit chunks; completion is never a blind skip.
        for arm in ARMS:
            for i,n,s in plans:
                d=out/arm/f'chunk_{i:03d}.npz'
                check_meta(json.loads(d.with_suffix('.json').read_text()),d,i,n,s,provenance)
        return
    with torch.inference_mode():
        teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
        reader,reader_config=load_result(READER,dev,'teacher_only')
        reader.eval().requires_grad_(False)
        if reader_config['checkpoint_sha256']!=READER_SHA: raise ValueError('Loader checkpoint mismatch')
        write_json(out/'reader_config.json',reader_config)
        vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
            local_files_only=True,torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
        for i,n,seed in plans:
            for arm in ARMS:
                folder=out/arm;folder.mkdir(exist_ok=True)
                dest=folder/f'chunk_{i:03d}.npz'; meta=dest.with_suffix('.json')
                if meta.exists():
                    check_meta(json.loads(meta.read_text()),dest,i,n,seed,provenance)
                    continue
                if shutil.disk_usage('.').free < 16*2**30: raise RuntimeError('Disk below16GiB safety floor')
                wrapper=LabelledRollout(teacher,None if arm=='original_rdd' else reader)
                z,_=sample_rdd(wrapper,n,[80,80,90],dev,seed)
                if wrapper.fine_calls!=90 or wrapper.reused_fine_calls!=(90 if arm=='reader' else 0):
                    raise ValueError('Wrong fine calls')
                endpoints={r:{k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                    for k,v in d.items()} for r,d in wrapper.endpoints.items()}
                if set(endpoints)!={'8','16'}: raise ValueError('Missing coarse endpoints')
                labels=wrapper.labels.tolist()
                if arm=='reader':
                    ref=json.loads((out/'original_rdd'/meta.name).read_text())
                    if ref['endpoints']!=endpoints or ref['labels']!=labels: raise ValueError('Unpaired original coarse/labels')
                if not torch.isfinite(z).all(): raise ValueError('Nonfinite latents')
                pics=torch.cat([decode(vae,z[j:j+4]) for j in range(0,n,4)])
                pixels=np.stack([np.asarray(to_pil(x)) for x in pics])
                if pixels.shape!=(n,256,256,3) or pixels.dtype!=np.uint8: raise ValueError('Wrong output pixels')
                with dest.with_suffix('.npz.part').open('wb') as f:
                    np.savez_compressed(f,arr_0=pixels,arr_1=np.asarray(labels,dtype=np.int64))
                os.replace(dest.with_suffix('.npz.part'),dest)
                if i==0:
                    for j,pic in enumerate(pics[:8]):to_pil(pic).save(folder/f'preview_{j:02d}.png')
                rec=dict(complete=True,index=i,count=n,seed=seed,provenance=provenance,labels=labels,
                         endpoints=endpoints,sha256=sha(dest),peak_gib=torch.cuda.max_memory_allocated()/2**30)
                write_json(meta,rec)
                print(json.dumps(dict(event='chunk_complete',rank=args.rank,arm=arm,batch=i+1,
                    batches=len(plans),count=n,seed=seed,peak_gib=rec['peak_gib'])),flush=True)
                del wrapper,z,pics,pixels
    write_json(out/'complete.json',dict(complete=True,**provenance))


def assemble(args):
    import numpy as np
    configs=[json.loads((args.root/f'shard_{r}/complete.json').read_text()) for r in range(4)]
    if any(c!=configs[0] or not c['complete'] for c in configs) or configs[0]['protocol']!=protocol(args.smoke):
        raise ValueError('Missing/mismatched shards')
    provenance={k:v for k,v in configs[0].items() if k!='complete'}
    total=protocol(args.smoke)['count']
    for arm in ARMS:
        dest=args.root/f'{arm}.npz';done=args.root/f'{arm}_assembled.json'
        if done.exists():
            d=json.loads(done.read_text())
            if d['provenance']!=provenance or d['sha256']!=sha(dest) or d['count']!=total:
                raise ValueError('Changed assembled output')
            continue
        if shutil.disk_usage('.').free < 18*2**30: raise RuntimeError('Need18GiB before final NPZ assembly')
        # ~1.84GiB RAM per10K arm. Do not create another multi-GB pixels.npy copy.
        arr=np.empty((total,256,256,3),dtype=np.uint8);labels=[];offset=0
        for rank in range(4):
            for i,n,seed in batches(rank,args.smoke):
                p=args.root/f'shard_{rank}'/arm/f'chunk_{i:03d}.npz'
                m=json.loads(p.with_suffix('.json').read_text())
                check_meta(m,p,i,n,seed,provenance)
                ref=json.loads((args.root/f'shard_{rank}'/'original_rdd'/p.with_suffix('.json').name).read_text())
                if m['labels']!=ref['labels'] or m['endpoints']!=ref['endpoints']: raise ValueError('Unpaired chunks')
                with np.load(p,allow_pickle=False) as d:
                    if d['arr_0'].shape!=(n,256,256,3) or d['arr_0'].dtype!=np.uint8 or d['arr_1'].tolist()!=m['labels']:
                        raise ValueError('Wrong chunk payload')
                    arr[offset:offset+n]=d['arr_0'];labels.extend(m['labels']);offset+=n
        if offset!=total: raise ValueError('Wrong sample count')
        with zipfile.ZipFile(dest.with_suffix('.npz.part'),'w',allowZip64=True) as z:
            with z.open('arr_0.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,arr,allow_pickle=False)
            with z.open('arr_1.npy','w',force_zip64=True) as f:np.lib.format.write_array(f,np.asarray(labels,dtype=np.int64),allow_pickle=False)
        del arr
        os.replace(dest.with_suffix('.npz.part'),dest)
        with np.load(dest,allow_pickle=False) as d:
            if d['arr_0'].shape!=(total,256,256,3) or d['arr_1'].tolist()!=labels: raise ValueError('Final NPZ verification failed')
        write_json(done,dict(complete=True,count=total,sha256=sha(dest),provenance=provenance))
    write_json(args.root/'sampling_complete.json',dict(complete=True,**provenance,
        warning='New fixed seeds independent of training/256preview; same-N10K estimate, NOT published50K FID or a latency benchmark.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('sample','assemble'))
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--rank',type=int,choices=range(4),default=0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    (sample if args.mode=='sample' else assemble)(args)
