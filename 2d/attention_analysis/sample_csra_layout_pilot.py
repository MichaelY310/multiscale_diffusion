#!/usr/bin/env python3
"""64 fixed paired images/arm after all four layout models reach800 updates.

Includes original teacher and prior1600-update best6 reference. The latter has
twice the new training budget, so it is NOT part of the equal-budget ablation.
No timestep-window switching: each new reader operates on all90 fine NFEs.
"""
import argparse
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights,resnet50
from csra_layout_student import LayoutStudent,FORMAT
from csra_layer_layouts import LAYOUTS,layout
from csra_skip_schedule import ScheduledPrefixRollout
from sample_csra_prefix_quality_pilot import load_student
from sample_csra_skip_schedules import sha,tensor_sha,ROOT,CKPT
from sample_csra_adapter_preview import sample_rdd,decode,save_grid,to_pil,AutoencoderKL,relative_l2
from train_rdd_csra_adapter import load_frozen_teacher
from train_csra_layout_pilot import hashes
from probe_csra_front_keep import BASE_SHA
from probe_csra_prefix_information import BASE,write_json


def load_layout(root,name,dev):
    path=Path(root)/'training'/f'{name}_aligned/final.pt'
    ck=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    c=ck['contract'];ip=Path(root)/'layouts'/name/'selected_ridge.pt'
    if (c['format']!=FORMAT or c['layout']!=layout(name) or c['base_sha256']!=BASE_SHA
        or c['coarse_source']!='aligned' or c['smoke'] or ck['step']!=800
        or c['initializer_sha256']!=sha(ip) or c['source_sha256']!=hashes()):
        raise ValueError('Wrong/incomplete layout checkpoint or changed source')
    init=torch.load(ip,map_location='cpu',weights_only=True)
    student=LayoutStudent(load_frozen_teacher(BASE,dev,'DiT-L/2','rdd'),init,name,
        checkpoint_blocks=False).to(dev)
    student.load_state_dict(ck['student']);student.eval().requires_grad_(False)
    return student,dict(checkpoint=str(path),checkpoint_sha256=sha(path),contract=c,validation=ck['validation'])


@torch.inference_mode()
def main(args):
    if args.rank not in range(4): raise ValueError('Four0-3 shards only')
    torch.set_num_threads(2);dev=torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0]<20*2**30: raise RuntimeError('Need20GiB free; do not stop other jobs')
    torch.cuda.set_per_process_memory_fraction(.18,dev);torch.backends.cuda.matmul.allow_tf32=False
    root=Path(args.root)/'layout_quality'/f'shard_{args.rank}';root.mkdir(parents=True,exist_ok=True)
    if (root/'metrics.json').exists():
        old=json.loads((root/'metrics.json').read_text())
        for row in old['rows']:
            if sha(root/row['variant']/f"{row['local_image']:03d}.png")!=row['image_sha256']:
                raise ValueError('Completed preview PNG changed')
        print('Completed preview hashes verified',flush=True);return
    seed=51560915+args.rank*10000;batch=16
    for mp in Path('eval_results/csra_rollout_data4k_20260912').glob('shard_*/manifest.json'):
        if any(r['seed']==seed for r in json.loads(mp.read_text())['files']): raise ValueError('Overlapping seed')
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    latents,configs,audits={},{},{}
    for name in ('original_dense','previous_prefix6_1600',*LAYOUTS):
        if name=='original_dense': student=None
        elif name=='previous_prefix6_1600': student,configs[name]=load_student('aligned',dev,CKPT,6,1600)
        else: student,configs[name]=load_layout(args.root,name,dev)
        wrapper=ScheduledPrefixRollout(teacher,student,range(90) if student is not None else ())
        z,_=sample_rdd(wrapper,batch,[80,80,90],dev,seed)
        if name=='original_dense': endpoints,labels=wrapper.endpoints,wrapper.labels
        torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
        difference=max(float((v.float()-endpoints[r][k].float()).abs().max()) for r,d in wrapper.endpoints.items() for k,v in d.items())
        if difference!=0 or set(wrapper.endpoints)!={'8','16'} or wrapper.fine_calls!=90:
            raise ValueError('Coarse endpoint/labels/fine-count parity failed')
        if student is not None and wrapper.reused_fine_calls!=90: raise ValueError('Unexpected reader activity')
        latents[name]=z.cpu()
        audits[name]=dict(coarse_endpoint_difference=difference,fine_calls=90,
            reader_active_fine_calls=wrapper.reused_fine_calls,latent_sha256=tensor_sha(z))
        torch.save(latents[name],root/f'{name}_latents.pt')
        write_json(root/'sampling_progress.json',audits)
        print(json.dumps(dict(rank=args.rank,sampled=name)),flush=True)
        del student,wrapper,z;torch.cuda.empty_cache()
    peak=torch.cuda.max_memory_allocated()/2**30
    del teacher;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures={}
    for name,z in latents.items():
        images=torch.cat([decode(vae,z[i:i+4].to(dev)) for i in range(0,batch,4)])
        pictures[name]=images.cpu();folder=root/name;folder.mkdir(exist_ok=True)
        for i,img in enumerate(images):to_pil(img).save(folder/f'{i:03d}.png')
        if name!='original_dense':
            save_grid([('original',pictures['original_dense'][:4]),(name,images[:4])],root/f'{name}_comparison.png')
    del vae,pictures;torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2
    cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier=resnet50(weights=None).eval();classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=weights.transforms();rows=[]
    for name in latents:
        for start in range(0,batch,8):
            inputs=[]
            for i in range(start,start+8):
                with Image.open(root/name/f'{i:03d}.png') as im:inputs.append(transform(im.convert('RGB')))
            probs=classifier(torch.stack(inputs)).softmax(-1)
            for j,p in enumerate(probs):
                i=start+j;target=int(labels[i]);ids=p.topk(5).indices.tolist()
                rows.append(dict(variant=name,local_image=i,image=args.rank*batch+i,target=target,
                    target_name=weights.meta['categories'][target],target_probability=float(p[target]),
                    top1_correct=ids[0]==target,top5_correct=target in ids,top5=ids,
                    image_sha256=sha(root/name/f'{i:03d}.png')))
    write_json(root/'metrics.json',dict(complete=True,seed=seed,images_per_arm=batch,labels=labels.tolist(),
        configs=configs,audits=audits,rows=rows,steps=[80,80,90],cfg=1.,boundary='noise_filling',
        peak_memory_gib=max(peak,torch.cuda.max_memory_allocated()/2**30),
        relative_latent_l2={k:relative_l2(z,latents['original_dense']) for k,z in latents.items()},
        warning='64 development pairs/arm, not FID or speed. Four new layouts each trained800 updates; previous_prefix6 trained1600, only historical reference. Equal6skipped/18executed blocks, but split3+3 pays for two complete readers. All stages8/16 unchanged and same labels/noise. Classifier accuracy is not perceptual quality.'))
    print(json.dumps(dict(rank=args.rank,complete=True)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--rank',type=int,required=True)
    main(p.parse_args())
