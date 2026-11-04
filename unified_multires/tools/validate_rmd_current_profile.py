#!/usr/bin/env python3
"""RMD published warmup/optimizer/sampling flags, optional full Wan architecture.

Read only the existing local Wan config; no weights or data downloads.
Video grid and CPU precision remain explicit conformance deviations.
"""
import argparse,json,hashlib,os,sys,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
import validate_rmd as subject
import conformance
from activation_trace import ActivationAudit
from umdiff import native
OUT=ROOT/'evidence/current_recipe_parity_20261009/components'

def main():
    global OUT
    p=argparse.ArgumentParser();p.add_argument('--full-backbone',action='store_true');p.add_argument('--steps',type=int,default=128);p.add_argument('--output',type=Path);p.add_argument('--transitions',default='1,2');p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    if args.output:OUT=args.output.resolve();conformance.ARTIFACT_ROOT=OUT/'artifacts'
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(args.threads);torch.manual_seed(20261009)
    config=ROOT/'sources/rmd/configs/train_wan_1_3b.yaml'
    import yaml
    recipe=yaml.safe_load(config.read_text());name='rmd_full_backbone' if args.full_backbone else 'rmd_released_flags'
    local=Path('/scr/yuheng/bugswarm/FastVideo/huggingface/hub/models--Wan-AI--Wan2.1-T2V-1.3B-Diffusers/snapshots/0fad780a534b6463e45facd96134c9f345acfa5b/transformer/config.json')
    if args.full_backbone:subject.MODEL={'name':'wan_transformer','args':{k:v for k,v in json.loads(local.read_text()).items() if not k.startswith('_')}}
    audits=[]
    def factory(label,pairs):
        audit=ActivationAudit(label,pairs,OUT/name);audits.append(audit);return audit
    subject.AUDIT_FACTORY=factory;subject.CHECKPOINT_ROOT=ROOT/'runs'/OUT.parent.name/OUT.name/name
    started=time.time();details=None;failure=None
    print(json.dumps(dict(event='start',profile=name,model=subject.MODEL,steps=args.steps)),flush=True)
    try:details=subject.train(args.steps,warmup_step=recipe['warmup_step'],target_size=(6,10),transitions=tuple(int(t) for t in args.transitions.split(',')),save_checkpoints=not args.full_backbone)
    except Exception:failure=traceback.format_exc();print(failure,flush=True)
    activation=[a.close() for a in audits]
    if any(a['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in a['totals'].values()) for a in activation):failure=failure or 'Activation/gradient bytes differ'
    report=dict(profile=name,status='failed' if failure else 'passed',device='cpu',gpu_used=False,steps=args.steps,
       full_released_architecture=args.full_backbone,actual_input_shape=[2,16 if args.full_backbone else 4,2,4,6],
       template_source=str(config.relative_to(ROOT)),template_sha256=hashlib.sha256(config.read_bytes()).hexdigest(),model=subject.MODEL,
       local_architecture_config_sha256=hashlib.sha256(local.read_bytes()).hexdigest() if args.full_backbone else None,
       full_paper_quality_verified=False,original_best_config_fully_executed=False,seconds=time.time()-started,details=details,failure=failure,
       activation=activation,checks=conformance.CHECKS,source_executions=native.AUDIT,
       deviations=['Random constructor weights and synthetic text; existing local asset used only to read architecture config.',
                   'CPU FP32 instead of CUDA BF16; video grid2x4x6 -> 2x6x10, original grid21x60x104 -> 21x90x160 not run.',
                   'Per-phase source optimizers/gradient clipping and published15-step warmup checked; distributed/accumulated training is a separate test.',
                   'Tiny random Wan VAE for decoder arithmetic; no pretrained video quality claim.'])
    (OUT/(name+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('profile','status','seconds','failure')}),flush=True)
    return bool(failure)
if __name__=='__main__':raise SystemExit(main())
