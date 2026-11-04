"""Paired256 development screen; reuse verified teacher-only control images.

Original RDD is rerun to check exact PNG and coarse-endpoint parity. No FID or
timing claims. All controls and fixed first8 grids retained, not cherry-picked.
"""
import argparse
import gc
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50

from csra_state_replay_data import ROOT as OLD_ROOT, QUALITY_SEED, PARENT_SHA
from train_csra_memory_refiner import ROOT, PROTOCOL, load_result
from sample_csra_prefix_quality_pilot import LabelledRollout
from sample_csra_adapter_preview import sample_rdd, decode, save_grid, to_pil, AutoencoderKL, relative_l2
from train_rdd_csra_adapter import load_frozen_teacher
from probe_csra_front_keep import BASE, BASE_SHA, file_sha, write_json
from summarize_csra_state_replay import paired


@torch.inference_mode()
def run(shard):
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in ('0', '1', '2', '3'):
        raise ValueError('Only single authorized physical GPU0-3')
    torch.set_num_threads(2)
    dev = torch.device('cuda:0')
    if torch.cuda.mem_get_info(dev)[0] < 18*2**30: raise RuntimeError('Need18GiB free')
    torch.cuda.set_per_process_memory_fraction(.18, dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    out = ROOT/'quality'/f'shard_{shard}'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'metrics.json').exists(): raise FileExistsError('Do not overwrite complete quality')
    old_path = OLD_ROOT/'quality'/f'shard_{shard}'
    old = json.loads((old_path/'metrics.json').read_text())
    seed = QUALITY_SEED+shard*10000
    if not old['complete'] or old['seed'] != seed or old['parent_sha256'] != PARENT_SHA:
        raise ValueError('Wrong paired control')
    control = old['configs']['teacher_only']
    if control['updates'] != 400 or file_sha(control['checkpoint']) != control['checkpoint_sha256']:
        raise ValueError('Changed matched400 teacher-only checkpoint')
    if file_sha(BASE) != BASE_SHA: raise ValueError('Changed original model')
    teacher = load_frozen_teacher(BASE, dev, 'DiT-L/2', 'rdd')
    original = LabelledRollout(teacher)
    z, _ = sample_rdd(original, 64, [80,80,90], dev, seed)
    labels, endpoints = original.labels, original.endpoints
    if labels.tolist() != old['labels']: raise ValueError('Unmatched labels')
    latents = {'original': z.cpu()}
    del z, original
    model, config = load_result(dev)
    wrapper = LabelledRollout(teacher, model)
    z, _ = sample_rdd(wrapper, 64, [80,80,90], dev, seed)
    torch.testing.assert_close(wrapper.labels, labels, atol=0, rtol=0)
    if wrapper.fine_calls != 90 or wrapper.reused_fine_calls != 90:
        raise ValueError('Wrong fine call count')
    if set(wrapper.endpoints) != set(endpoints): raise ValueError('Missing coarse endpoints')
    for r, values in endpoints.items():
        for k, v in values.items(): torch.testing.assert_close(wrapper.endpoints[r][k], v, atol=0, rtol=0)
    latents['refined'] = z.cpu()
    del teacher, model, wrapper, z
    gc.collect(); torch.cuda.empty_cache()
    vae = AutoencoderKL.from_pretrained('checkpoints/pretrained/stabilityai/sd-vae-ft-ema',
        local_files_only=True, torch_dtype=torch.bfloat16).to(dev).eval().requires_grad_(False)
    pictures = {}
    lookup = {(r['variant'], r['local_image']): r for r in old['rows']}
    for name, z in latents.items():
        pictures[name] = torch.cat([decode(vae, z[i:i+4].to(dev)) for i in range(0,64,4)])
        (out/name).mkdir(exist_ok=True)
        for i, pic in enumerate(pictures[name]):
            path = out/name/f'{i:03d}.png'; to_pil(pic).save(path)
            if name == 'original' and file_sha(path) != lookup[name,i]['image_sha256']:
                raise ValueError('Original PNG changed; controls no longer paired exactly')
    (out/'teacher_only').mkdir(exist_ok=True)
    controls = []
    for i in range(64):
        path = old_path/'teacher_only'/f'{i:03d}.png'
        if file_sha(path) != lookup['teacher_only',i]['image_sha256']: raise ValueError('Control image changed')
        shutil.copy2(path, out/'teacher_only'/path.name)
        if i < 8:
            with Image.open(path) as im:
                controls.append(torch.from_numpy(np.array(im.convert('RGB'))).permute(2,0,1).float()/255)
    save_grid([('original', pictures['original'][:8]), ('teacher_only', torch.stack(controls)),
               ('refined', pictures['refined'][:8])], out/'comparison_grid.png')
    peak = torch.cuda.max_memory_allocated()/2**30
    del vae, pictures; gc.collect(); torch.cuda.empty_cache()
    cp = Path(torch.hub.get_dir())/'checkpoints/resnet50-11ad3fa6.pth'
    if file_sha(cp) != old['classifier_sha256']: raise ValueError('Classifier changed')
    classifier = resnet50(weights=None).eval()
    classifier.load_state_dict(torch.load(cp, map_location='cpu', weights_only=True))
    transform = ResNet50_Weights.IMAGENET1K_V2.transforms()
    rows = [r for r in old['rows'] if r['variant'] in ('original', 'teacher_only')]
    for start in range(0,64,8):
        images = []
        for i in range(start,start+8):
            with Image.open(out/'refined'/f'{i:03d}.png') as im: images.append(transform(im.convert('RGB')))
        probs = classifier(torch.stack(images)).softmax(-1)
        for j, prob in enumerate(probs):
            i = start+j; target = int(labels[i]); top = prob.topk(5).indices.tolist()
            rows.append(dict(image=shard*64+i,local_image=i,variant='refined',target=target,
                top1_correct=top[0]==target,top5_correct=target in top,target_probability=float(prob[target]),
                image_sha256=file_sha(out/'refined'/f'{i:03d}.png')))
    write_json(out/'metrics.json',dict(complete=True,protocol=PROTOCOL,shard=shard,seed=seed,
        labels=labels.tolist(),steps=[80,80,90],cfg=1.,boundary='noise_filling',updates=400,
        config=config,control=control,control_metrics_sha256=file_sha(old_path/'metrics.json'),
        classifier_sha256=file_sha(cp),coarse_endpoint_max_abs=0.,original_png_parity=True,
        rows=rows,peak_gib=peak,relative_latent_l2=dict(original=0.,teacher_only=old['relative_latent_l2']['teacher_only'],
            refined=relative_l2(latents['refined'],latents['original'])),
        warning='Reused256 development samples and control; one training seed, not held-out test/FID/video quality.'))


def summarize():
    rows=[]; metrics=[]
    for shard in range(4):
        p=ROOT/'quality'/f'shard_{shard}'
        d=json.loads((p/'metrics.json').read_text())
        if (not d['complete'] or d['protocol']!=PROTOCOL or d['seed']!=QUALITY_SEED+shard*10000
            or d['coarse_endpoint_max_abs']!=0 or not d['original_png_parity']): raise ValueError('Bad shard')
        if len(d['rows'])!=192: raise ValueError('Missing rows')
        for r in d['rows']:
            if file_sha(p/r['variant']/f"{r['local_image']:03d}.png")!=r['image_sha256']: raise ValueError('Image changed')
        rows.extend(d['rows']); metrics.append(d)
    lookup={(r['image'],r['variant']):r for r in rows}
    if len(lookup)!=768: raise ValueError('Duplicated rows')
    variants=('original','teacher_only','refined')
    fields=('top1_correct','top5_correct','target_probability')
    agg={v:{k:float(np.mean([r[k] for r in rows if r['variant']==v])) for k in fields} for v in variants}
    for v in variants: agg[v]['relative_latent_l2']=float(np.mean([d['relative_latent_l2'][v] for d in metrics]))
    contrasts={b:{k:paired([float(lookup[i,'refined'][k])-float(lookup[i,b][k]) for i in range(256)]) for k in fields}
               for b in ('teacher_only','original')}
    write_json(ROOT/'quality_summary.json',dict(complete=True,protocol=PROTOCOL,images_sha_verified=768,
        aggregates=agg,refined_minus=contrasts,warning='Development screening, one seed, not FID or success certification.'))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shard',type=int,choices=range(4))
    p.add_argument('--summarize',action='store_true')
    args=p.parse_args()
    if args.summarize: summarize()
    elif args.shard is not None: run(args.shard)
    else: p.error('Need --shard or --summarize')
