"""Fixed400 updates, fresh256 development images x4variants; no FID/speed claim."""
import argparse
import gc
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from csra_state_replay_data import ROOT, QUALITY_SEED, PROTOCOL, PARENT, PARENT_SHA
from train_csra_state_replay import make_model, load_result
from sample_csra_prefix_quality_pilot import LabelledRollout
from sample_csra_adapter_preview import sample_rdd, decode, save_grid, to_pil, AutoencoderKL, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher
from probe_csra_front_keep import BASE, BASE_SHA, file_sha, write_json


@torch.inference_mode()
def run(args):
    torch.set_num_threads(2)
    dev=torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0]<18*2**30:raise RuntimeError('Need18GiB free')
    torch.cuda.set_per_process_memory_fraction(.18,dev)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    out=ROOT/'quality'/f'shard_{args.shard}';out.mkdir(parents=True,exist_ok=True)
    if (out/'metrics.json').exists():raise FileExistsError('Do not overwrite completed quality')
    seed=QUALITY_SEED+args.shard*10000
    for mp in (ROOT/'data').glob('shard_*/manifest.json'):
        if any(f['seed']==seed for f in json.loads(mp.read_text())['files']):raise ValueError('Quality seed overlap')
    if file_sha(BASE)!=BASE_SHA:raise ValueError('Changed teacher')
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    baseline=LabelledRollout(teacher)
    z,_=sample_rdd(baseline,64,[80,80,90],dev,seed)
    latents={'original':z.cpu()};labels=baseline.labels;endpoints=baseline.endpoints
    configs={};parity={}
    del baseline,z
    for name in ('parent','teacher_only','mixed50'):
        if name=='parent':model,configs[name]=make_model(dev)
        else:
            path=Path(f'checkpoints/RDD-CSRA-state-replay-{name}-L400K-20260914/latest.pt')
            model,configs[name]=load_result(path,dev,name)
        wrapper=LabelledRollout(teacher,model)
        z,_=sample_rdd(wrapper,64,[80,80,90],dev,seed)
        torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
        if wrapper.fine_calls!=90 or wrapper.reused_fine_calls!=90:raise ValueError('Wrong call count')
        if set(wrapper.endpoints)!=set(endpoints):raise ValueError('Missing original coarse endpoints')
        for r,values in endpoints.items():
            for key,v in values.items():torch.testing.assert_close(v,wrapper.endpoints[r][key],atol=0,rtol=0)
        parity[name]=0.;latents[name]=z.cpu()
        print(json.dumps(dict(event='sampled',shard=args.shard,variant=name)),flush=True)
        del model,wrapper,z;gc.collect();torch.cuda.empty_cache()
    del teacher
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True,torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures={}
    for name,z in latents.items():
        pictures[name]=torch.cat([decode(vae,z[i:i+4].to(dev)) for i in range(0,64,4)])
        folder=out/name;folder.mkdir(exist_ok=True)
        for i,pic in enumerate(pictures[name]):to_pil(pic).save(folder/f'{i:03d}.png')
    save_grid([(name,v[:8]) for name,v in pictures.items()],out/'comparison_grid.png')
    peak=torch.cuda.max_memory_allocated()/2**30
    del vae,pictures;gc.collect();torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2
    cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    if not cp.is_file():raise FileNotFoundError('Need cached classifier; no download')
    model=resnet50(weights=None).eval()
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=weights.transforms();rows=[]
    for name in latents:
        for start in range(0,64,8):
            images=[]
            for i in range(start,start+8):
                with Image.open(out/name/f'{i:03d}.png') as im:images.append(transform(im.convert('RGB')))
            probabilities=model(torch.stack(images)).softmax(-1)
            for j,prob in enumerate(probabilities):
                i=start+j;target=int(labels[i]);top=prob.topk(5)
                rows.append(dict(image=args.shard*64+i,local_image=i,variant=name,target=target,
                    top1_correct=int(top.indices[0])==target,top5_correct=target in top.indices.tolist(),
                    target_probability=float(prob[target]),image_sha256=file_sha(out/name/f'{i:03d}.png')))
    write_json(out/'metrics.json',dict(complete=True,protocol=PROTOCOL,shard=args.shard,seed=seed,
        steps=[80,80,90],cfg=1.,boundary='noise_filling',parent=PARENT,parent_sha256=PARENT_SHA,
        updates=400,labels=labels.tolist(),coarse_endpoint_max_abs=parity,configs=configs,
        rows=rows,images_per_variant=64,peak_gib=peak,
        relative_latent_l2={name:relative_l2(v,latents['original']) for name,v in latents.items()},
        classifier='ResNet50 ImageNet1K V2 CPUFP32 official transforms',classifier_sha256=file_sha(cp),
        warning='Fresh256 development images, one training seed. NOT FID/realism/video quality or timing. '
                'Original versus unchanged parent and two fixed400 continuations. Snapshot replay, not live on-policy.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',type=int,choices=range(4),required=True)
    run(p.parse_args())
