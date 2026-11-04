"""Small readout-gate ablation, NOT a speed benchmark. No weight changes.

Keep spatially aligned coarse affine path and current-fine MLP unchanged.
Only multiply native-coarse attention read_out by1/.5/0 in both readers.
The attention kernel still executes even for0: never claim saved FLOPs here.
"""
import argparse
import json
import os
from pathlib import Path
import torch
from torchvision.models import ResNet50_Weights,resnet50
from sample_csra_split_quality256 import load as load800
from sample_csra_split1600_quality256 import load as load1600
from sample_csra_adapter_preview import sample_rdd,decode,save_grid,to_pil,AutoencoderKL
from csra_skip_schedule import ScheduledPrefixRollout
from sample_csra_skip_schedules import sha,ROOT
from probe_csra_prefix_information import BASE,write_json
from train_rdd_csra_adapter import load_frozen_teacher


@torch.inference_mode()
def main(a):
    if a.rank not in range(4) or os.environ.get('CUDA_VISIBLE_DEVICES')!=str(a.rank):raise ValueError('GPU0-3 exact shard')
    dev=torch.device('cuda:0');torch.set_num_threads(2)
    if torch.cuda.mem_get_info(dev)[0]<20*2**30:raise RuntimeError('Need20GiB free')
    torch.cuda.set_per_process_memory_fraction(.20,dev);torch.backends.cuda.matmul.allow_tf32=False
    step=800 if a.rank<2 else 1600;seed=88660915+(a.rank%2)*10000;batch=16
    out=Path(a.root)/'attention_gate_pilot'/f'shard_{a.rank}';out.mkdir(parents=True,exist_ok=True)
    protocol=dict(version='csra_readout_gate_quality_v1',step=step,seed=seed,rank=a.rank,batch=batch,
        gates={'original':None,'gate1':1.,'gate05':.5,'gate0':0.},script_sha256=sha(__file__),
        steps=[80,80,90],cfg=1.,boundary='noise_filling',
        warning='32 pairs/parent, exploratory. Attention still runs under gate0; NOT a runtime optimization benchmark.')
    if (out/'protocol.json').exists() and json.loads((out/'protocol.json').read_text())!=protocol:raise ValueError('Different existing protocol')
    if (out/'metrics.json').exists():return
    write_json(out/'protocol.json',protocol)
    teacher=load_frozen_teacher(BASE,dev,'DiT-L/2','rdd')
    student,config=(load800 if step==800 else load1600)(a.root,'split3_3','aligned',dev)
    latents={}
    for name,gate in protocol['gates'].items():
        handles=[]
        if gate is not None and gate!=1.:
            handles=[reader.read_out.register_forward_hook(lambda module,args,out,g=gate:out*g) for reader in student.readers]
        try:
            wrapper=ScheduledPrefixRollout(teacher,None if gate is None else student,() if gate is None else range(90))
            z,_=sample_rdd(wrapper,batch,[80,80,90],dev,seed)
        finally:
            for handle in handles:handle.remove()
        if name=='original':endpoints,labels=wrapper.endpoints,wrapper.labels
        torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
        for res,d in wrapper.endpoints.items():
            for key,value in d.items():torch.testing.assert_close(value,endpoints[res][key],atol=0,rtol=0)
        latents[name]=z.cpu();del z,wrapper
        print(json.dumps(dict(rank=a.rank,step=step,sampled=name)),flush=True)
    del teacher,student;torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',local_files_only=True,
        torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures={}
    for name,z in latents.items():
        pictures[name]=torch.cat([decode(vae,z[i:i+4].to(dev)) for i in range(0,batch,4)]).cpu()
        folder=out/name;folder.mkdir(exist_ok=True)
        for i,img in enumerate(pictures[name]):to_pil(img).save(folder/f'{i:03d}.png')
    save_grid([(name,images[:8]) for name,images in pictures.items()],out/'comparison.png')
    del vae;torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2;cp=Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    classifier=resnet50(weights=None).eval();classifier.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True))
    transform=weights.transforms();rows=[]
    for name,images in pictures.items():
        for start in range(0,batch,8):
            probs=classifier(torch.stack([transform(to_pil(im)) for im in images[start:start+8]])).softmax(-1)
            for j,p in enumerate(probs):
                i=start+j;target=int(labels[i]);top=p.topk(5).indices.tolist()
                rows.append(dict(variant=name,image=(a.rank%2)*batch+i,local_image=i,target=target,
                    top1_correct=top[0]==target,top5_correct=target in top,target_probability=float(p[target]),
                    image_sha256=sha(out/name/f'{i:03d}.png')))
    write_json(out/'metrics.json',dict(complete=True,protocol=protocol,config=config,rows=rows,
        labels=labels.tolist(),coarse_endpoint_max_abs=0.,classifier_sha256=sha(cp),
        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=ROOT);p.add_argument('--rank',type=int,required=True)
    main(p.parse_args())
