#!/usr/bin/env python3
"""VAE invocation/parameter flags/RNG ordering at native microbatch boundaries."""
import argparse,copy,hashlib,json,os,random,sys,tempfile,time,traceback
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build,native
from umdiff.engine import Method
from umdiff.input_transforms import LocalVAE,VAEMoments
from umdiff.training import Trainer
from umdiff.randomness import GlobalRandomStream
from validate_training import nonzero_output
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from activation_trace import ActivationAudit,byte_view
from compare_released_profiles import compare_state


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(731);np.random.seed(731);random.seed(731);started=time.time();audit=None;report=dict(gpu_used=False,formal_training=False)
    try:
        from diffusers import AutoencoderKL
        from umdiff.source_objectives import setup
        c=json.loads((ROOT/'configs/native_fixtures/ppflow_2.json').read_text());c['model']['args']['class_dropout_prob']=.1
        recipe=dict(lr=1e-4,ema_decay=.9999,accumulation_steps=2,optimizer_parameters='all',randomness='native_global')
        vae=AutoencoderKL(block_out_channels=(8,)*4,down_block_types=('DownEncoderBlock2D',)*4,
            up_block_types=('UpDecoderBlock2D',)*4,latent_channels=4,layers_per_block=1,norm_num_groups=4).eval()
        with tempfile.TemporaryDirectory(prefix='umdiff_codec_boundary_') as tmp:
            vae.save_pretrained(tmp,safe_serialization=True)
            codec=LocalVAE({'name':'local_autoencoder','args':{'path':tmp}},freeze_parameters=False)
            net=nonzero_output(build('model',c['model']));trainer=Trainer(Method(c),copy.deepcopy(net),input_transform=codec,**recipe)
            opt=torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0.);ema=copy.deepcopy(net).eval().requires_grad_(False)
            images=torch.randn(16,3,128,128);labels=torch.arange(16)
            audit=ActivationAudit('online_vae_microbatch',[('network',net,trainer.model),('vae',vae,codec.model)],a.output/'layers');rows=[]
            for step in range(4):
                audit.begin(step);audit.side('native');before=snapshot_rng();net.train();opt.zero_grad(set_to_none=True)
                for lo in (0,8):
                    x=images[lo:lo+8]
                    with torch.no_grad():z=native.statements('ppflow/train.py',322,322,dict(vae=vae,x=x))['x']
                    _,original,_,_,_=setup('ppflow',c,z,labels[lo:lo+8]);original(net,step).mean().div(2).backward()
                raw={k:None if q.grad is None else q.grad.detach().clone() for k,q in net.named_parameters()};opt.step()
                with torch.no_grad():
                    for q,v in zip(ema.parameters(),net.parameters()):q.mul_(.9999).add_(v,alpha=.0001)
                after=snapshot_rng();audit.side('unified');restore_rng(before)
                def gate(model):
                    for key,q in model.named_parameters():
                        expected=raw[key]
                        if (expected is None)!=(q.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(q.grad))):raise AssertionError('Raw codec-boundary gradient '+key)
                trainer.step(images,None,model_kwargs={'y':labels},before_gradient_processing=gate)
                audit.parameters(net,trainer.model);audit.finish_segment()
                checks={key:compare_state(u.state_dict(),v.state_dict()) for key,u,v in [('model',net,trainer.model),('optimizer',opt,trainer.optimizer),('ema',ema,trainer.ema)]}
                if any(v['exact']!=v['tensors'] for v in checks.values()) or fingerprint(after)!=fingerprint(snapshot_rng()):raise AssertionError('Microbatch state/RNG mismatch')
                rows.append(dict(step=step,state=checks,raw_gradients_exact=True,rng_exact=True))
            act=audit.close();audit=None
            if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('Microbatch layer/gradient mismatch')
            # Numerical negative controls: same mathematical linear map changes
            # rounding when PyTorch selects a frozen-parameter CPU fast path.
            linear=torch.nn.Linear(8,8);frozen=copy.deepcopy(linear).requires_grad_(False)
            value=torch.randn(8,8,16,16).view(8,8,256).transpose(1,2)
            with torch.no_grad():delta=float((linear(value)-frozen(value)).abs().max());frozen.requires_grad_(True);fixed=torch.equal(byte_view(linear(value)),byte_view(frozen(value)))
            if not delta or not fixed:raise AssertionError('Frozen-flag negative control not reproduced')
            moments=torch.randn(8,8,8,8);sampler=native.method('dimr/libs/autoencoder.py','sample',dict(torch=torch))
            pre=snapshot_rng();expected=torch.cat([sampler(SimpleNamespace(scale_factor=.18215),moments[lo:lo+4]) for lo in (0,4)])
            expected_rng=snapshot_rng();restore_rng(pre);actual=torch.cat([VAEMoments()(moments[lo:lo+4],GlobalRandomStream()) for lo in (0,4)])
            if not torch.equal(byte_view(actual),byte_view(expected)) or fingerprint(expected_rng)!=fingerprint(snapshot_rng()):raise AssertionError('Native moments/micro RNG mismatch')
            report.update(status='passed',steps=rows,activation=act,posterior_scope='Two native8-sample VAE calls per logical16-sample update;posterior and transport time/noise remain interleaved;no global encode-before-slicing',
                frozen_parameter_negative_control=dict(max_abs=delta,original_flags_restored_byte_exact=fixed),moments_microbatch_exact=True,
                scope='Actual PPFlow VAE assignment and source transport/loss against unified microbatch codec/Method/Trainer;fourupdates,random reduced source network+real four-level tiny VAE,CPU FP32. Logical accumulation overlay is explicit;official source has one local batch per update.')
    except Exception:report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
    if audit is not None:report['activation']=audit.close()
    report.update(seconds=time.time()-started,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_executions=native.AUDIT,cuda_initialized=torch.cuda.is_initialized())
    (a.output/'codec_boundaries.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','seconds')}),flush=True);return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
