#!/usr/bin/env python3
"""Fresh256 paired development samples plus equal-budget no-coarse control.

Selection used the previous64-image pilot. These are new seeds, but still a
development evaluation, NOT FID or video-quality evidence. All images are saved.
"""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights,resnet50
from csra_layout_student import LayoutStudent,FORMAT
from csra_layer_layouts import layout
from csra_skip_schedule import ScheduledPrefixRollout
from sample_csra_prefix_quality_pilot import load_student
from sample_csra_skip_schedules import sha,tensor_sha,ROOT,CKPT
from sample_csra_adapter_preview import sample_rdd,decode,save_grid,to_pil,AutoencoderKL,relative_l2
from train_rdd_csra_adapter import load_frozen_teacher
from train_csra_layout_pilot import hashes
from probe_csra_front_keep import BASE_SHA
from probe_csra_prefix_information import BASE,write_json

ARMS=('original_dense','previous_prefix6_1600','front0_6_aligned','split3_3_aligned','split3_3_zero')


def load(root,name,source,device):
    path=Path(root)/'training'/f'{name}_{source}/final.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    c=ck['contract'];ip=Path(root)/'layouts'/name/'selected_ridge.pt'
    if (c['format']!=FORMAT or c['layout']!=layout(name) or c['base_sha256']!=BASE_SHA
        or c['coarse_source']!=source or c['smoke'] or ck['step']!=800
        or c['initializer_sha256']!=sha(ip) or c['source_sha256']!=hashes()):
        raise ValueError('Wrong800-update checkpoint, source or changed training code')
    init=torch.load(ip,map_location='cpu',weights_only=True)
    model=LayoutStudent(load_frozen_teacher(BASE,device,'DiT-L/2','rdd'),init,name,source,
        checkpoint_blocks=False).to(device)
    model.load_state_dict(ck['student']);model.eval().requires_grad_(False)
    return model,dict(checkpoint=str(path),checkpoint_sha256=sha(path),contract=c,validation=ck['validation'])


@torch.inference_mode()
def sample(args):
    if args.rank not in range(4): raise ValueError('Four0-3 shards only')
    torch.set_num_threads(2);dev=torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0]<20*2**30: raise RuntimeError('Need20GiB free; other jobs untouched')
    torch.cuda.set_per_process_memory_fraction(.18,dev);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.root)/'split_quality256'/f'shard_{args.rank}';root.mkdir(parents=True,exist_ok=True)
    seed,batch=61560915+args.rank*10000,64
    protocol=dict(version='split3_3_fresh256_control_v1',rank=args.rank,seed=seed,batch=batch,arms=list(ARMS),
        script_sha256=sha(__file__),steps=[80,80,90],cfg=1.,boundary='noise_filling',base_sha256=BASE_SHA)
    if (root/'protocol.json').exists() and json.loads((root/'protocol.json').read_text())!=protocol:
        raise ValueError('Do not overwrite a different protocol')
    write_json(root/'protocol.json',protocol)
    if (root/'metrics.json').exists():
        print('Completed shard exists; full PNG/provenance audit runs after all four shards',flush=True);return
    if sha(BASE)!=BASE_SHA: raise ValueError('Base changed')
    for mp in Path('eval_results/csra_rollout_data4k_20260912').glob('shard_*/manifest.json'):
        if any(r['seed']==seed for r in json.loads(mp.read_text())['files']): raise ValueError('Overlapping seed')
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    latents,configs,audits={},{},{}
    for arm in ARMS:
        if arm=='original_dense': student=None
        elif arm=='previous_prefix6_1600':student,configs[arm]=load_student('aligned',dev,CKPT,6,1600)
        else:
            name,source=arm.rsplit('_',1);student,configs[arm]=load(args.root,name,source,dev)
        wrapper=ScheduledPrefixRollout(teacher,student,range(90) if student is not None else ())
        z,_=sample_rdd(wrapper,batch,[80,80,90],dev,seed)
        if arm=='original_dense': endpoints,labels=wrapper.endpoints,wrapper.labels
        torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
        difference=max(float((v.float()-endpoints[r][k].float()).abs().max()) for r,d in wrapper.endpoints.items() for k,v in d.items())
        if difference!=0 or set(wrapper.endpoints)!={'8','16'} or wrapper.fine_calls!=90:
            raise ValueError('Coarse/labels/fine-count parity failed')
        if student is not None and wrapper.reused_fine_calls!=90: raise ValueError('Reader was not active on all fine calls')
        latents[arm]=z.cpu()
        audits[arm]=dict(coarse_endpoint_difference=difference,fine_calls=90,
            reader_active_fine_calls=wrapper.reused_fine_calls,latent_sha256=tensor_sha(z))
        torch.save(latents[arm],root/f'{arm}_latents.pt')
        write_json(root/'sampling_progress.json',audits)
        print(json.dumps(dict(rank=args.rank,sampled=arm)),flush=True)
        del student,wrapper,z;torch.cuda.empty_cache()
    peak=torch.cuda.max_memory_allocated()/2**30
    del teacher;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures={}
    for arm,z in latents.items():
        images=torch.cat([decode(vae,z[i:i+4].to(dev)) for i in range(0,batch,4)])
        pictures[arm]=images.cpu();folder=root/arm;folder.mkdir(exist_ok=True)
        for i,img in enumerate(images):to_pil(img).save(folder/f'{i:03d}.png')
        if arm!='original_dense':
            save_grid([('original',pictures['original_dense'][:8]),(arm,images[:8])],root/f'{arm}_comparison.png')
    save_grid([('original',pictures['original_dense'][:8]),('split_aligned',pictures['split3_3_aligned'][:8]),
        ('split_zero',pictures['split3_3_zero'][:8])],root/'source_control_comparison.png')
    del vae,pictures;torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2
    cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier=resnet50(weights=None).eval();classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=weights.transforms();rows=[]
    for arm in ARMS:
        for start in range(0,batch,8):
            inputs=[]
            for i in range(start,start+8):
                with Image.open(root/arm/f'{i:03d}.png') as im:inputs.append(transform(im.convert('RGB')))
            probs=classifier(torch.stack(inputs)).softmax(-1)
            for j,p in enumerate(probs):
                i=start+j;target=int(labels[i]);ids=p.topk(5).indices.tolist()
                rows.append(dict(variant=arm,local_image=i,image=args.rank*batch+i,target=target,
                    target_name=weights.meta['categories'][target],target_probability=float(p[target]),
                    top1_correct=ids[0]==target,top5_correct=target in ids,top5=ids,
                    image_sha256=sha(root/arm/f'{i:03d}.png')))
    write_json(root/'metrics.json',dict(complete=True,protocol=protocol,seed=seed,images_per_arm=batch,labels=labels.tolist(),
        configs=configs,audits=audits,rows=rows,steps=[80,80,90],cfg=1.,boundary='noise_filling',
        classifier_sha256=sha(cp),peak_memory_gib=max(peak,torch.cuda.max_memory_allocated()/2**30),
        relative_latent_l2={k:relative_l2(z,latents['original_dense']) for k,z in latents.items()},
        warning='256 fresh development pairs per arm, not FID/perceptual-quality or speed. Equal800-update front0/split/aligned/zero controls. Prior-prefix6 has1600 updates, historical reference only. All original8/16 frozen. Two complete readers, not one, are required by split3+3.'))
    print(json.dumps(dict(rank=args.rank,complete=True)),flush=True)


def audit(args):
    folder=Path(args.root)/('layout_quality' if args.action=='audit64' else 'split_quality256')
    n=16 if args.action=='audit64' else 64
    records=[json.loads((folder/f'shard_{r}/metrics.json').read_text()) for r in range(4)]
    arms=tuple(records[0]['audits']);seen=set();rows=[];configurations={}
    for rank,data in enumerate(records):
        if not data['complete'] or data['images_per_arm']!=n or set(data['audits'])!=set(arms):
            raise ValueError('Incomplete/inconsistent shard')
        if data['seed'] in seen:raise ValueError('Duplicate seed')
        seen.add(data['seed'])
        if data['steps']!=[80,80,90] or data['cfg']!=1. or data['boundary']!='noise_filling':raise ValueError('Changed sampler')
        for name,rec in data['audits'].items():
            if rec['coarse_endpoint_difference']!=0 or rec['fine_calls']!=90: raise ValueError('Failed parity')
        for name,c in data['configs'].items():
            if name in configurations and configurations[name]!=c:raise ValueError('Different checkpoints between shards')
            configurations[name]=c
        for row in data['rows']:
            if row['target']!=data['labels'][row['local_image']] or row['image']!=rank*n+row['local_image']:
                raise ValueError('Labels/image pairing mismatch')
            path=folder/f'shard_{rank}'/row['variant']/f"{row['local_image']:03d}.png"
            if sha(path)!=row['image_sha256']:raise ValueError('PNG hash mismatch')
            rows.append(row)
    for name,c in configurations.items():
        if 'checkpoint_sha256' in c and sha(c['checkpoint'])!=c['checkpoint_sha256']:raise ValueError('Checkpoint changed')
        if 'contract' in c and c['contract']['source_sha256']!=hashes():raise ValueError('Training source changed')
    total=4*n;means={}
    for arm in arms:
        rr=[r for r in rows if r['variant']==arm]
        if len(rr)!=total or len({r['image'] for r in rr})!=total:raise ValueError('Wrong count/duplicate image')
        means[arm]=dict(n=total,top1=sum(r['top1_correct'] for r in rr)/total,
            top5=sum(r['top5_correct'] for r in rr)/total,
            target_probability=sum(r['target_probability'] for r in rr)/total,
            mean_relative_latent_l2=sum(d['relative_latent_l2'][arm] for d in records)/4)
    write_json(folder/'summary_and_audit.json',dict(complete=True,means=means,
        verified_pngs=len(rows),configurations=configurations,
        shard_sha256={str(i):sha(folder/f'shard_{i}/metrics.json') for i in range(4)},
        warning=records[0]['warning']))
    print(json.dumps(means,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=('sample','audit','audit64'))
    p.add_argument('--root',default=ROOT);p.add_argument('--rank',type=int)
    a=p.parse_args();sample(a) if a.action=='sample' else audit(a)
