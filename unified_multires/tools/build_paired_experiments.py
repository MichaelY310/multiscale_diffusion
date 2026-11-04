#!/usr/bin/env python3
"""Record published recipes separately from matched, source-core experiments."""
import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'configs/paired_experiments'

# None means unreported/unidentified, never a default guessed from another paper.
PAPERS = {
 'relay': dict(title='Relay Diffusion', arxiv='2309.03350', dataset='ImageNet-256; CelebA-HQ-256', batch=4096, optimizer='Adam', lr=1e-4, budget='ImageNet: 1250M training images across the cascade; per-stage allocation not identified', initialization='scratch, two separately trained models', evaluation='ImageNet FID 1.87; exact low/high checkpoint pair required', gaps=['Per-stage best-run image budget not identified; image-count EMA and loss scaling must be retained.']),
 'ssd': dict(title='Scale Space Diffusion', arxiv='2603.08709', dataset='CelebA-64/128/256; ImageNet-64', batch=64, optimizer='AdamW', lr=1e-4, budget='Released64 checkpoints: 1M updates; CelebA128/256: 300k', initialization='scratch Flexi-UNet', evaluation='2/4 levels at64,3/5 at128,3/6 at256; source DDPM sampler', gaps=['Batch64 is train_ssd.sh:32 per rank times2; best paper row is not uniquely selected.','Do not use lr_anneal_steps as a stop cap: it changes the LR schedule.']),
 'pixelflow': dict(title='PixelFlow: Pixel-Space Generative Models with Flow', arxiv='2504.07963', dataset='ImageNet-256 class conditional', batch=None, optimizer='AdamW', lr=1e-4, budget=None, initialization='scratch XL, raw pixels, patch4/head72', evaluation='FID1.98: Dopri5, atol1e-6, rtol.001 in code, four-stage CFG max2.4', gaps=['Public YAML is 10 epochs and batch4 per rank, not an identified best training run.','Published best training global batch/budget not identified in the inspected paper/release.']),
 'dfm': dict(title='Improving Progressive Generation with Decomposable Flow Matching', arxiv='2506.19839', dataset='ImageNet-512/1024; Kinetics-700', batch=256, optimizer='Adam (paper)', lr=1e-4, budget='512:500k;1024:350k;video:200k; warmup10k', initialization='Paper DiT-XL + scale-equivariant FLUX VAE; public release SiT + SD VAE', evaluation='Paper40 steps:30+10; thresholds.7/.95; CFG sweep', gaps=['Public SiT release explicitly differs from the paper architecture and VAE.','SE autoencoder/private fine-tuning data and original full best-run recipe are unavailable.']),
 'lapflow': dict(title='Laplacian Multi-scale Flow Matching for Generative Modeling', arxiv='2602.19461', dataset='ImageNet-256; CelebA-HQ-256/512/1024', batch=256, optimizer='AdamW', lr=1e-4, budget='ImageNetXL:120epochs; B:1400epochs; HQ:500/1000/2000epochs', initialization='ImageNet EQVAE; HQ256 EQVAE,512/1024 SDVAE', evaluation='ImageNet XL T=.5, EMA.9999; exact best CFG not identified', gaps=['CLI defaults T=.9/LR5e-4/EMA.9 differ from AppendixG.','HQ uses cosine2e-4 to1e-6, batches256/256/128,3scales at512/1024.','Released square sampler bug and exact best CFG remain unresolved.']),
 'ppflow': dict(title='Pyramidal Patchification Flow for Visual Generation', arxiv='2506.23543', dataset='ImageNet-256/512; mixed text-to-image datasets', batch=256, optimizer='AdamW', lr=1e-4, budget='XL256 pretrained:1M updates; XL512:400k; B scratch:11M; FLUX:90k', initialization='XL256 starts from SiT-XL/2; SD VAE', evaluation='XL2 FID1.99, CFG[1,3]; XL3 FID2.23, CFG[1,3.5,3.75]; source Heun2', gaps=['Public CLI defaults1400epochs and lacks an exact max-update/accumulation flag.','FLUX text-to-image and some paper variants are not released.','The SiT initialization includes substantial prior training cost.']),
 'dctdiff': dict(title='DCTdiff: Intriguing Properties of Image Generative Modeling in the DCT Space', arxiv='2412.15032', dataset='CIFAR10,CelebA64,ImageNet64,FFHQ128/256/512,AFHQ512', batch='256; ImageNet64:1024;512:128', optimizer='AdamW', lr='2e-4; ImageNet64:3e-4;512:1e-4', budget='Per-dataset released Python configs; CelebA64:250k;FFHQ128/256:300k', initialization='scratch U-ViT; dataset-specific DCT/EBFR statistics', evaluation='Native DPM/EM and exact DCT codec, resize and EBFR', gaps=['Paper DiT branches are not released.','A different dataset requires re-estimated DCT statistics, not copying CIFAR constants.']),
 'mdm': dict(title='Matryoshka Diffusion Models', arxiv='2310.15111', dataset='Released weights:50M Flickr text-image pairs; configs:CC12M', batch=None, optimizer='Adam', lr=5e-5, budget='Public CC12M config:1M updates; warmup10k; full Flickr curriculum unidentified', initialization='pretrained vision64/256; Flan-T5-XL;1024 inner256 checkpoint', evaluation='Nested256/1024;250 resampled steps, CFG7.5, clipping in demo', gaps=['Flickr training data/curriculum are not reproduced by CC12M configs.','Published best global batch not established; commented batch fields are not authoritative.','Online T5/streaming/DDP curriculum parity is unverified.']),
 'dimr': dict(title='Alleviating Distortion in Image Generation via Multi-Resolution Diffusion Models and Time-Dependent Layer Normalization', arxiv='2406.09416', dataset='ImageNet-256/512', batch=1024, optimizer='AdamW betas(.99,.99),wd.03', lr=2e-4, budget='Best released256 weights:800epochs; Python config:1M updates; warmup5k', initialization='scratch DiMR-G/2R orXL/2R; original SD VAE moments', evaluation='G/2R FID1.63; XL/2R1.70;250steps, native guidance scale.4,power-1', gaps=['800epochs and1M updates are different budgets; do not label the latter best-run reproduction.','Source fp16/accelerate16rank lifecycle and real VAE pipeline are not GPU-certified here.']),
 'diffusability': dict(title='Improving the Diffusability of Autoencoders', arxiv='2502.14831', dataset='ImageNet;Kinetics700; additional codec fine-tuning data', batch=None, optimizer=None, lr=None, budget=None, initialization='pretrained codec, SE2/4 fine-tuning, downstream generator retraining', evaluation='Codec SE/VGG loss plus separately retrained generator quality', gaps=['Release README still lists training/evaluation instructions and checkpoints as TODO.','Codec component checks cannot reproduce the full generator-quality main table.']),
 'rmd': dict(title='Cross-Resolution Distribution Matching for Diffusion Distillation', arxiv='2603.06136', dataset='Video Wan14B; image SDXL/PixArt/SD3.5', batch='Paper video4; image32; release14B64', optimizer='AdamW, beta1=0,beta2=.95(video)/.999(image)', lr='generator5e-6(video)/4e-6(image); fake2e-5', budget='Release14B:max300,warmup15; best paper budget not established', initialization='pretrained teacher/student/fake models; two-phase distillation', evaluation='Native6-step rollout, CFG5, eta.5, flowshift3; optional GAN separate', gaps=['Public14B effective batch=4x8x2=64, not paper video batch4.','Local full14B/GAN/main-table recipe not certified; source author attribution remains unconfirmed.']),
 'speed': dict(title='SPEED: Spectral Progressive Evolution for Efficient Diffusion', arxiv='2605.18736', dataset='Text-to-image FLUX.1-dev1024 benchmarks', batch='inference batch; no training batch', optimizer=None, lr=None, budget='training-free inference', initialization='pretrained FLUX, no training', evaluation='Spectral transform/DCT-Haar-FFT, scale schedule and NFE must match each row', gaps=['Not a training baseline; needs exact pretrained weights, prompts, metric assets and selected main-table row.']),
 'nami': dict(title='NAMI: Efficient Image Generation with Progressive Multi-Scale Diffusion', arxiv='2503.09242', dataset='Text-to-image, FLUX-derived progressive model', batch='Paper training768; public inference only', optimizer=None, lr=None, budget='No released training entrypoint', initialization='trained NAMI transformer and learned bridge, vendor Diffusers', evaluation='Progressive depth/resolution and trained bridge; distinct from plain FLUX', gaps=['Published training recipe cannot be recovered from an inference-only release.']),
 'mrflow': dict(title='MrFlow: Multi-resolution Flow for Efficient Image Generation', arxiv='2607.01642', dataset='Text-to-image FLUX/Qwen1024 benchmarks', batch='inference batch; no training batch', optimizer=None, lr=None, budget='training-free inference', initialization='pretrained FLUX/Qwen plus exact RealESRGAN', evaluation='Low-res flow, PIL/VAE/SR transition, high-res refinement; exact NFE/model required', gaps=['Not a scratch-training baseline; evaluate exact full pretrained pipeline and transitions.']),
 'rdd': dict(title='RDD: migrated local implementation', arxiv=None, dataset='ImageNet-256 class conditional', batch=256, optimizer='AdamW', lr=1e-4, budget='Historical L reached492000; XL407600; scripts epoch ceiling1400', initialization='scratch DiT-L/2 orXL/2; SD VAE ema;32/16/8 latent grids', evaluation='300/300/400 stages; EMA.9999; retained source native staged sampler', gaps=['400k is a newly chosen common budget, not the exact last step of both historical runs.','Historical image datasets/weights are not all migrated; do not resume deleted GPU-check payloads.']),
}

FAIR = dict(dataset='ImageNet-1K/train', resolution=256, task='class_conditional',
 global_batch=256, world_size=1, micro_batch=16, accumulation=16,
 updates=400000, seed=0, optimizer=dict(optimizer_name='adamw',lr=1e-4,
 betas=[.9,.999],eps=1e-8,weight_decay=0.,ema_decay=.9999,
 precision='float32',grad_clip=None,lr_schedule=None,randomness='native_global'))

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def profile_dataset(paper,name):
    # This describes the selected release profile, not every task in a paper.
    if paper=='dctdiff':
        key=name.removeprefix('dctdiff_').split('_uvit')[0]
        return {'cifar10':'CIFAR-10 32x32','imgnet64':'ImageNet 64x64','celeba64':'CelebA 64x64','afhq512':'AFHQ 512x512','ffhq128':'FFHQ 128x128','ffhq256':'FFHQ 256x256','ffhq512':'FFHQ 512x512'}[key]
    if paper in {'dfm','dimr','ppflow','lapflow','rdd','pixelflow'}:return 'ImageNet 256x256 (selected released image profile)'
    if paper=='relay':return 'ImageNet '+('64x64' if name.endswith('_low') else '256x256')+' native ZIP'
    if paper=='ssd':return 'CelebA 64x64, official train split required'
    if paper=='mdm':return 'CC12M released '+('1024' if name.endswith('1024') else '256')+' YAML; not the Flickr50M checkpoint run'
    if paper=='rmd':return 'Prompt-only input; exact released prompt corpus must be bound, not a real-video dataset'
    if paper=='diffusability':return 'Unbound dataset: SE2/4 codec component, no identified best training run'
    raise ValueError(paper)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    pins=json.loads((ROOT/'evidence/sources.json').read_text())
    index=json.loads((ROOT/'configs/paper_recipes/INDEX.json').read_text())
    if isinstance(index,dict):index=index['recipes']
    profiles=[]
    names={'dctdiff':'dctdiff','dimr':'dimr','pixelflow':'pixelflow','ppflow':'ppflow','dfm':'dfm','lapflow':'lapflow','ssd':'ssd','mdm':'mdm','rdd':'rdd','relay':'relay','diffusability':'diffusability','rmd':'rmd'}
    for entry in index:
        name=entry['name'];paper=next(value for prefix,value in names.items() if name.startswith(prefix+'_'))
        c=json.loads((ROOT/entry['path']).read_text())
        eligible=paper in {'rdd','ppflow','dimr','pixelflow','dfm','lapflow'}
        if name=='dfm_sit_xl_repa_localfeatures':eligible=False
        if paper=='lapflow' and name!='lapflow_imagenet_xl_2_paper':eligible=False
        seed=0
        if paper=='dctdiff':seed=123456 if 'imgnet64' in name else 1234
        elif paper=='dimr':seed=1234
        elif paper=='pixelflow':seed=42
        elif paper=='lapflow':seed=1024
        elif paper in ('relay','ssd','mdm','diffusability','rmd'):seed=None
        contract=dict(dataset=profile_dataset(paper,name),shape=c.get('image_shape'),
          global_batch=c['evidence'].get('author_global_batch'),budget=c['evidence'].get('author_steps'),
          optimizer=c['training']['optimizer'],seed=seed,world_size=1,
          implementation_scope='shared lifecycle + pinned author source core; NOT unmodified author CLI')
        path=OUT/(name+'.json');path.write_text(json.dumps(c,indent=2)+'\n')
        profiles.append(dict(id=name,paper=paper,config=path.name,config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
          template=entry['path'],fair_eligible=eligible,paper_contract=contract,
          full_paper_result_certified=False,author_cli_lifecycle_certified=False))
    # The historical L run is explicit, rather than silently selecting only XL.
    xl=next(p for p in profiles if p['id']=='rdd_unified_xl_300_300_400')
    lc=json.loads((OUT/xl['config']).read_text());lc['model']['args'].update(hidden_size=1024,depth=24)
    ln='rdd_unified_l_300_300_400';lp=OUT/(ln+'.json');lp.write_text(json.dumps(lc,indent=2)+'\n')
    le=copy.deepcopy(xl);le.update(id=ln,config=lp.name,config_sha256=hashlib.sha256(lp.read_bytes()).hexdigest());profiles.append(le)
    # Paper Table3/4 uses the released Dopri5 branch, unlike the Euler demo.
    pf=next(p for p in profiles if p['paper']=='pixelflow')
    pc=json.loads((OUT/pf['config']).read_text());pc['sampler']['args']['solver']='native_dopri5'
    pn='pixelflow_xl_c2i_paper_dopri5';pp=OUT/(pn+'.json');pp.write_text(json.dumps(pc,indent=2)+'\n')
    pe=copy.deepcopy(pf);pe.update(id=pn,config=pp.name,config_sha256=hashlib.sha256(pp.read_bytes()).hexdigest());profiles.append(pe)
    for paper,row in PAPERS.items():
        if paper in pins:row['repository']={k:pins[paper][k] for k in ('repo','commit')}
        row['paper_url']='https://arxiv.org/html/'+row['arxiv'] if row['arxiv'] else None
    coverage=json.loads((ROOT/'PAPER_COVERAGE.json').read_text())
    skipped=[]
    for entry in coverage['methods']:
        if entry['id'] in PAPERS:
            PAPERS[entry['id']]['title']=entry['name']
        else:
            skipped.append(dict(id=entry['id'],title=entry['name'],paper_url=entry['paper'],
                reason=entry.get('skip_reason','No executable original release certified; literature mapping only.')))
    manifest=dict(version=1,created='2026-10-10',full_training_started=False,
      fair_rdd=FAIR,papers=PAPERS,profiles=profiles,unreleased=skipped,coverage_reference='PAPER_COVERAGE.json',
      policies=dict(default_action='plan',download_models=False,download_data=False,
        gpu_policy='Previous authorization expired; CPU only until explicit renewal with deadline.',
        best_result_guaranteed=False,shared_frontend_is_author_cli=False,
        limitations=['Single-process matched-core runner; multi-rank author lifecycle is not certified.',
          'Fair-rdd preserves each architecture/codec/initialization/target/path/sampler; not a common-backbone ablation.',
          'Equal updates/global batch imply equal images, NOT equal FLOPs or pretrained cost.',
          'Small FP32 synthetic checks cannot guarantee full-training FID, BF16/CUDA/DDP equivalence.']))
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    assets=dict(data={},assets={},note='Bind local real dataset paths and exact author codec/initialization paths; nothing is downloaded.')
    (OUT/'assets.example.json').write_text(json.dumps(assets,indent=2)+'\n')
    print(json.dumps(dict(profiles=len(profiles),papers=len(PAPERS),manifest=str(OUT/'manifest.json'))))

if __name__=='__main__':main()
