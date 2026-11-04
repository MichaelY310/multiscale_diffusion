#!/usr/bin/env python3
"""Bounded, paired screening of WHEN to activate the fixed best prefix6 reader.

64 development images per arm, no FID claim, no video timing extrapolation.
All inactive fine calls execute the ORIGINAL full teacher. No new training here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from csra_skip_schedule import schedules, ScheduledPrefixRollout
from sample_csra_prefix_quality_pilot import load_student, LabelledRollout
from sample_csra_adapter_preview import sample_rdd, decode, save_grid, to_pil, AutoencoderKL, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher
from probe_csra_prefix_information import BASE, write_json

ROOT = '/home/yuheng/rdd_experiments/csra_skip_layout_20260915'
CKPT = 'checkpoints/RDD-CSRA-prefix6-conditional1600-aligned-L400K-20260913/best.pt'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


@torch.inference_mode()
def sample(args):
    if args.rank not in range(4): raise ValueError('Use physical GPUs0-3 only')
    torch.set_num_threads(2)
    root = Path(args.root)/'schedules'/f'shard_{args.rank}'
    root.mkdir(parents=True, exist_ok=True)
    seed, batch = 41560915 + 10000*args.rank, 16
    contract = dict(version='prefix6_schedule_screen_v1', seed=seed, batch=batch,
        schedules={k:list(v) for k,v in schedules().items()}, checkpoint=CKPT,
        checkpoint_sha256=sha(CKPT), source_sha256={str(p):sha(p) for p in [
            Path(__file__), Path(__file__).with_name('csra_skip_schedule.py'),
            Path(__file__).with_name('csra_prefix_student.py')]})
    manifest = root/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('Existing experiment protocol changed; use a new output root')
    write_json(manifest, contract)
    if (root/'metrics.json').exists():
        for row in json.loads((root/'metrics.json').read_text())['rows']:
            if sha(root/row['variant']/f"{row['local_image']:03d}.png") != row['image_sha256']:
                raise ValueError('Completed PNG mismatch')
        print(f'{root}: already complete and PNG hashes checked', flush=True); return
    if shutil.disk_usage(root).free < 8*2**30: raise RuntimeError('Need8GiB free; never delete data')
    for mp in Path('eval_results/csra_rollout_data4k_20260912').glob('shard_*/manifest.json'):
        if any(item['seed'] == seed for item in json.loads(mp.read_text())['files']):
            raise ValueError('Seed overlaps saved training/development corpus')
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 20*2**30: raise RuntimeError('Need20GiB free; leave other jobs alone')
    torch.cuda.set_per_process_memory_fraction(.18, dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    teacher = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    student, config = load_student('aligned', dev, CKPT, 6, 1600)
    latents, audits = {}, {}
    original_endpoints, labels = None, None
    for name, indices in schedules().items():
        wrapper = ScheduledPrefixRollout(teacher, student if indices else None, indices)
        begin = time.monotonic()
        z, _ = sample_rdd(wrapper, batch, [80,80,90], dev, seed)
        if wrapper.fine_calls != 90 or wrapper.reused_fine_calls != len(indices):
            raise ValueError('Unexpected actual call counts')
        if name == 'original_dense':
            original_endpoints, labels = wrapper.endpoints, wrapper.labels
        torch.testing.assert_close(wrapper.labels, labels, atol=0, rtol=0)
        if set(wrapper.endpoints) != {'8','16'}: raise ValueError('Missing coarse endpoints')
        difference = max(float((v.float()-original_endpoints[r][k].float()).abs().max())
            for r,d in wrapper.endpoints.items() for k,v in d.items())
        if difference != 0: raise ValueError('Original coarse trajectory changed')
        if name == 'all90' and args.rank == 0:
            reference = LabelledRollout(teacher, student)
            expected, _ = sample_rdd(reference, batch, [80,80,90], dev, seed)
            torch.testing.assert_close(z, expected, atol=0, rtol=0)
            del reference, expected
        latents[name] = z.cpu()
        torch.save(latents[name], root/f'{name}_latents.pt')
        audits[name] = dict(fine_calls=wrapper.fine_calls, reader_calls=wrapper.reused_fine_calls,
            active_global_timesteps=wrapper.active_timesteps, global_timesteps=wrapper.fine_timesteps,
            endpoint_difference=difference, latent_sha256=tensor_sha(z),
            diagnostic_wall_seconds=time.monotonic()-begin)
        write_json(root/'sampling_progress.json', audits)
        print(json.dumps(dict(rank=args.rank, sampled=name, reader_calls=len(indices))), flush=True)
        del wrapper, z
    peak = torch.cuda.max_memory_allocated()/2**30
    del teacher, student
    torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True, torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures = {}
    for name, z in latents.items():
        images = torch.cat([decode(vae, z[i:i+4].to(dev)) for i in range(0,batch,4)])
        folder = root/name; folder.mkdir(exist_ok=True)
        for i,img in enumerate(images): to_pil(img).save(folder/f'{i:03d}.png')
        pictures[name] = images.cpu()
        if name != 'original_dense':
            save_grid([('original', pictures['original_dense'][:4]), (name, images[:4])],
                root/f'{name}_comparison.png')
    del vae, pictures
    torch.cuda.empty_cache()
    weights = ResNet50_Weights.IMAGENET1K_V2
    cp = Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier = resnet50(weights=None).eval()
    classifier.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True))
    transform = weights.transforms(); rows=[]
    for name in latents:
        for start in range(0,batch,8):
            inputs=[]
            for i in range(start,min(start+8,batch)):
                with Image.open(root/name/f'{i:03d}.png') as im: inputs.append(transform(im.convert('RGB')))
            probs=classifier(torch.stack(inputs)).softmax(-1)
            for j,p in enumerate(probs):
                i=start+j; target=int(labels[i]); ids=p.topk(5).indices.tolist()
                rows.append(dict(variant=name, local_image=i, image=args.rank*batch+i,
                    target=target, target_name=weights.meta['categories'][target],
                    target_probability=float(p[target]), top1_correct=ids[0]==target,
                    top5_correct=target in ids, top5=ids,
                    image_sha256=sha(root/name/f'{i:03d}.png')))
    write_json(root/'metrics.json', dict(complete=True, protocol=contract, config=config,
        labels=labels.tolist(), audits=audits, rows=rows, steps=[80,80,90], cfg=1.,
        boundary='noise_filling', coarse_endpoint_sha256={f'{r}/{k}':tensor_sha(v)
            for r,d in original_endpoints.items() for k,v in d.items()},
        all90_legacy_bitwise_parity=True if args.rank==0 else 'checked on shard0',
        relative_latent_l2={k:relative_l2(z,latents['original_dense']) for k,z in latents.items()},
        peak_memory_gib=max(peak,torch.cuda.max_memory_allocated()/2**30),
        warning='64 development images/arm across4shards, not FID, not final test, not speed. Inactive fineNFEs use original full teacher; quality changes confound time placement and switching fine weightsets. FastWan DMD3NFE has only one fineNFE: these schedules cannot be claimed as video speed improvements.'))
    print(json.dumps(dict(rank=args.rank, complete=True)), flush=True)


def summarize(args):
    root=Path(args.root)/'schedules'
    shards=[json.loads((root/f'shard_{i}/metrics.json').read_text()) for i in range(4)]
    if not all(s['complete'] for s in shards): raise RuntimeError('Incomplete shards')
    if not shards[0]['all90_legacy_bitwise_parity']: raise RuntimeError('Missing old-runtime parity')
    rows=[r for s in shards for r in s['rows']]
    means={}
    for name, indices in schedules().items():
        selected=[r for r in rows if r['variant']==name]
        if len(selected)!=64: raise ValueError('Expected64 paired images per arm')
        means[name]=dict(samples=64, reader_calls=len(indices),
            top1=sum(r['top1_correct'] for r in selected)/64,
            top5=sum(r['top5_correct'] for r in selected)/64,
            target_probability=sum(r['target_probability'] for r in selected)/64,
            relative_latent_l2_mean=sum(s['relative_latent_l2'][name] for s in shards)/4)
    write_json(root/'summary.json', dict(complete=True, means=means,
        warning=shards[0]['warning'], shard_metrics_sha256={str(i):sha(root/f'shard_{i}/metrics.json') for i in range(4)}))
    print(json.dumps(means, indent=2), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=('sample','summarize'))
    p.add_argument('--root',default=ROOT); p.add_argument('--rank',type=int)
    a=p.parse_args(); sample(a) if a.action=='sample' else summarize(a)
