"""Opt-in matched 10K sample pilot; old checkpoints/sampling code are unchanged.

Four independent GPU workers, sequential arms, bounded batches and resumable
atomic chunks. Assembly verifies labels, original coarse parity and provenance.
Quality only: do not interpret these instrumented run times as a speed test.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import zipfile

ARMS = ('original_dense', 'aligned', 'zero')
TEMPLATE = 'checkpoints/RDD-CSRA-prefix6-conditional1600-{arm}-L400K-20260913/best.pt'


def batches(rank, smoke=False):
    if rank not in range(4):
        raise ValueError('Only ranks 0-3')
    count = 4 if smoke else 2500
    return [(i, min(64, count-i*64), 20560914+rank*100000+i*1000)
            for i in range((count+63)//64)]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()


def write_json(path, value):
    tmp = path.with_suffix(path.suffix+'.part')
    tmp.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    os.replace(tmp, path)


def protocol(smoke):
    return dict(version=1, smoke=smoke, samples_per_arm=16 if smoke else 10000,
                arms=list(ARMS), prefix_depth=6, checkpoint_step=1600,
                steps=[80,80,90], cfg=1., upsample='noise_filling', tf32=False,
                vae='sd-vae-ft-ema/bfloat16/scale0.18215', decode_batch=4,
                seed_base=20560914, batch_size=64)


def sample(args):
    import numpy as np
    import torch
    from sample_csra_prefix_quality_pilot import LabelledRollout, load_student, BASE
    from sample_csra_adapter_preview import sample_rdd, decode, to_pil, AutoencoderKL
    from train_rdd_csra_adapter import load_frozen_teacher
    torch.set_num_threads(2)
    device = torch.device('cuda:0')
    if torch.cuda.mem_get_info()[0] < 20*2**30:
        raise RuntimeError('Need20GiB free; no other process will be stopped')
    # A hard allocator ceiling for this process, not a claim of GPU exclusivity.
    torch.cuda.set_per_process_memory_fraction(.18, device)
    torch.backends.cuda.matmul.allow_tf32 = False
    out = args.root/f'shard_{args.rank}'
    out.mkdir(parents=True, exist_ok=True)
    plan = batches(args.rank, args.smoke)
    seeds = {s for _,_,s in plan}
    for mp in Path('eval_results/csra_rollout_data4k_20260912').glob('shard_*/manifest.json'):
        if any(row['seed'] in seeds for row in json.loads(mp.read_text())['files']):
            raise ValueError('Seed overlaps training corpus')
    provenance = dict(protocol=protocol(args.smoke), base_sha256=sha(BASE),
                      checkpoints={a:sha(TEMPLATE.format(arm=a)) for a in ARMS[1:]})
    config_file = out/'config.json'
    if config_file.exists() and json.loads(config_file.read_text()) != provenance:
        raise ValueError('Refuse mixing checkpoint/configuration versions')
    write_json(config_file, provenance)
    with torch.inference_mode():
        teacher = load_frozen_teacher(BASE, device, 'DiT-L/2', 'rdd')
        vae = AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
            local_files_only=True, torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
        for arm in ARMS:
            folder = out/arm
            folder.mkdir(exist_ok=True)
            model = None
            if arm != 'original_dense':
                model, _ = load_student(arm, device, TEMPLATE, 6, 1600)
            for index, count, seed in plan:
                dest = folder/f'chunk_{index:03d}.npz'
                meta = dest.with_suffix('.json')
                if meta.exists():
                    previous = json.loads(meta.read_text())
                    if previous['seed'] != seed or previous['count'] != count or sha(dest) != previous['sha256']:
                        raise ValueError('Corrupt completed chunk')
                    continue
                wrapper = LabelledRollout(teacher, model)
                z, _ = sample_rdd(wrapper, count, [80,80,90], device, seed)
                if wrapper.fine_calls != 90 or wrapper.reused_fine_calls != (0 if model is None else 90):
                    raise ValueError('Wrong fine invocation count')
                endpoints = {r:{k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                for k,v in rec.items()} for r,rec in wrapper.endpoints.items()}
                if set(endpoints) != {'8','16'}: raise ValueError('Missing endpoints')
                labels = wrapper.labels.tolist()
                if arm != 'original_dense':
                    ref = json.loads((out/'original_dense'/meta.name).read_text())
                    if ref['endpoints'] != endpoints or ref['labels'] != labels:
                        raise ValueError('Original coarse path or labels changed')
                if not torch.isfinite(z).all(): raise ValueError('Nonfinite latent')
                ims = torch.cat([decode(vae,z[i:i+4]) for i in range(0,count,4)])
                pixels = np.stack([np.asarray(to_pil(x)) for x in ims])
                with dest.with_suffix('.npz.part').open('wb') as f:
                    np.savez_compressed(f, arr_0=pixels, arr_1=np.asarray(labels,dtype=np.int64))
                os.replace(dest.with_suffix('.npz.part'),dest)
                if index == 0:
                    for i,x in enumerate(ims[:8]): to_pil(x).save(folder/f'preview_{i:02d}.png')
                record = dict(complete=True,seed=seed,count=count,labels=labels,endpoints=endpoints,
                              sha256=sha(dest),peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                write_json(meta,record)
                print(json.dumps(dict(rank=args.rank,arm=arm,batch=index+1,batches=len(plan),**record)),flush=True)
                del wrapper,z,ims,pixels
            del model
            torch.cuda.empty_cache()
    write_json(out/'complete.json',dict(complete=True,**provenance))


def assemble(args):
    import numpy as np
    configs = [json.loads((args.root/f'shard_{r}/complete.json').read_text()) for r in range(4)]
    if not all(c == configs[0] and c['complete'] for c in configs):
        raise ValueError('Incomplete or mismatched shards')
    if configs[0]['protocol'] != protocol(args.smoke): raise ValueError('Wrong requested protocol')
    n = protocol(args.smoke)['samples_per_arm']
    for arm in ARMS:
        path = args.root/f'{arm}.npz'
        data_path = args.root/f'{arm}_pixels.npy'
        arr = np.lib.format.open_memmap(data_path,mode='w+',dtype=np.uint8,shape=(n,256,256,3))
        labels = []; offset = 0
        for rank in range(4):
            for idx,count,seed in batches(rank,args.smoke):
                chunk = args.root/f'shard_{rank}'/arm/f'chunk_{idx:03d}.npz'
                m = json.loads(chunk.with_suffix('.json').read_text())
                ref = json.loads((args.root/f'shard_{rank}/original_dense'/chunk.with_suffix('.json').name).read_text())
                if (m['count']!=count or m['seed']!=seed or m['sha256']!=sha(chunk)
                    or m['labels']!=ref['labels'] or m['endpoints']!=ref['endpoints']):
                    raise ValueError('Unpaired/corrupt batch')
                with np.load(chunk) as v:
                    if v['arr_0'].dtype!=np.uint8 or v['arr_0'].shape!=(count,256,256,3): raise ValueError('Invalid pixels')
                    if v['arr_1'].tolist()!=m['labels']: raise ValueError('Invalid labels')
                    arr[offset:offset+count]=v['arr_0'];labels.extend(m['labels'])
                offset+=count
        assert offset==n
        arr.flush();del arr
        label_path=args.root/f'{arm}_labels.npy';np.save(label_path,np.asarray(labels,dtype=np.int64))
        with zipfile.ZipFile(path.with_suffix('.npz.part'),'w',allowZip64=True) as z:
            z.write(data_path,'arr_0.npy');z.write(label_path,'arr_1.npy')
        os.replace(path.with_suffix('.npz.part'),path)
        # Retain recoverable chunks and source arrays; no cleanup of user data.
        write_json(args.root/f'{arm}_assembled.json',dict(complete=True,count=n,sha256=sha(path)))
    write_json(args.root/'sampling_complete.json',dict(**configs[0],
        warning='10K development FID pilot, not comparable to published50K FID; same reference and count for all three arms.'))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['sample','assemble'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--rank',type=int,default=0)
    p.add_argument('--smoke',action='store_true')
    args=p.parse_args()
    (sample if args.mode=='sample' else assemble)(args)
