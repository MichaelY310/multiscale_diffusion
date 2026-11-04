"""Diagnostic only: restore all eight fine blocks in the trained backbone.

This is not accelerated inference. Restored blocks retain original frozen
weights, while embeddings/suffix were trained with the bridge. Failure of this
untrained composition does NOT identify a single cause of quality loss.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from sample_csra_prefix_quality_pilot import LabelledRollout, load_student
from sample_csra_adapter_preview import sample_rdd, manual_forward, decode, relative_l2, save_grid, to_pil, AutoencoderKL
from train_rdd_csra_adapter import load_frozen_teacher
from probe_csra_prefix_information import BASE, write_json


class BypassRollout(LabelledRollout):
    def forward(self, x, t, y):
        if int(x.shape[-1]) == 32:
            self.fine_calls += 1
            # Trained patch/time/class embeddings + frozen first8 blocks +
            # trained last16 blocks/final head. No bridge output is mixed in.
            return manual_forward(self.student.backbone, x, t, y, multires=True)[0]
        return super().forward(x, t, y)


@torch.inference_mode()
def main(seed):
    root=Path('eval_results/csra_prefix8_bypass_20260913')/f'seed_{seed}'
    if (root/'metrics.json').exists():
        raise FileExistsError('Already complete; do not rerun')
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    device=torch.device('cuda:0')
    if torch.cuda.mem_get_info(device)[0]<18*2**30:
        raise RuntimeError('Need18GiB free; never stop another job')
    torch.backends.cuda.matmul.allow_tf32=False
    teacher=load_frozen_teacher(BASE, device, 'DiT-L/2', 'rdd')
    original=LabelledRollout(teacher)
    z,_=sample_rdd(original,4,[80,80,90],device,seed)
    latents={'original_dense':z.cpu()};configs={};parity={};calls={}
    labels=original.labels
    for arm in ('aligned','zero'):
        student,configs[arm]=load_student(arm, device)
        for mode,klass in (('bridge',LabelledRollout),('restore8',BypassRollout)):
            key=f'{arm}_{mode}'
            wrapper=klass(teacher,student)
            z,_=sample_rdd(wrapper,4,[80,80,90],device,seed)
            torch.testing.assert_close(wrapper.labels,labels,atol=0,rtol=0)
            if wrapper.fine_calls!=90:raise ValueError('Wrong fine calls')
            expected=90 if mode=='bridge' else 0
            if wrapper.reused_fine_calls!=expected:raise ValueError('Wrong replacement calls')
            if set(wrapper.endpoints)!={'8','16'}:raise ValueError('Missing coarse endpoints')
            delta=max(float((v.float()-original.endpoints[r][k].float()).abs().max())
                for r,record in wrapper.endpoints.items() for k,v in record.items())
            if delta!=0:raise ValueError('Original coarse path changed')
            parity[key]=delta;calls[key]=dict(fine=90,bridge=expected)
            latents[key]=z.cpu()
            print(json.dumps(dict(seed=seed,phase='sampled',variant=key)),flush=True)
            del wrapper,z
        del student
        torch.cuda.empty_cache()
    del original,teacher
    vae=AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True,torch_dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    pictures={};hashes={}
    for variant,z in latents.items():
        pictures[variant]=decode(vae,z.to(device))
        folder=root/variant;folder.mkdir(exist_ok=True)
        hashes[variant]=[]
        for i,img in enumerate(pictures[variant]):
            file=folder/f'{i:03d}.png';to_pil(img).save(file)
            hashes[variant].append(hashlib.sha256(file.read_bytes()).hexdigest())
            refkey={'aligned_bridge':'prefix_aligned','zero_bridge':'prefix_zero'}.get(variant,variant)
            if variant in ('original_dense','aligned_bridge','zero_bridge'):
                ref=Path('eval_results/csra_prefix8_conditional3200_pair_20260913')/f'seed_{seed}'/refkey/f'{i:03d}.png'
                if not ref.exists() or hashlib.sha256(ref.read_bytes()).hexdigest()!=hashes[variant][-1]:
                    raise ValueError('Original/bridge image differs from completed3200 preview')
    save_grid(list(pictures.items()),root/'comparison_grid.png')
    peak=torch.cuda.max_memory_allocated()/2**30
    del pictures,vae
    torch.cuda.empty_cache()
    weights=ResNet50_Weights.IMAGENET1K_V2
    cache=Path(torch.hub.get_dir())/'checkpoints'/'resnet50-11ad3fa6.pth'
    if not cache.exists():raise FileNotFoundError('Classifier must already be cached')
    classifier=resnet50(weights=None).eval()
    classifier.load_state_dict(torch.load(cache,map_location='cpu',weights_only=True))
    transform=weights.transforms();names=weights.meta['categories'];rows=[]
    for variant in latents:
        batch=[]
        for i in range(4):
            with Image.open(root/variant/f'{i:03d}.png') as im:batch.append(transform(im.convert('RGB')))
        probabilities=classifier(torch.stack(batch)).softmax(-1)
        for i,prob in enumerate(probabilities):
            target=int(labels[i]);top=prob.topk(5)
            rows.append(dict(image=i,variant=variant,target=target,target_name=names[target],
                top1_correct=int(top.indices[0])==target,top5_correct=target in top.indices.tolist(),
                target_probability=float(prob[target]),image_sha256=hashes[variant][i]))
    write_json(root/'metrics.json',dict(complete=True,seed=seed,configs=configs,labels=labels.tolist(),
        rows=rows,coarse_parity=parity,calls=calls,peak_memory_gib=peak,
        relative_latent_l2={k:relative_l2(v,latents['original_dense']) for k,v in latents.items()},
        steps=[80,80,90],cfg=1.,upsample='noise_filling',
        warning='Diagnostic, NOT accelerated inference. Both restore8 variants execute all24 fine blocks, with trained embeddings/suffix but frozen original first8. This composition was NOT jointly trained, so failure cannot prove bridge-independent degradation. Twelve reused development examples only, not FID or quality validation. Original/coarse path and all original/bridge reference PNG hashes exact.'))
    print(json.dumps(dict(complete=True,seed=seed,peak_memory_gib=peak)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True)
    args=p.parse_args()
    if args.seed not in (20260921,20260923,20275004):raise ValueError('Fixed diagnostic seeds only')
    main(args.seed)
