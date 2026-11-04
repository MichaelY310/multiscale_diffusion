#!/usr/bin/env python3
"""CPU FP16 GradScaler/overflow lifecycle; not CUDA kernel certification."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build
from umdiff.engine import Method
from umdiff.training import Trainer
from umdiff.randomness import GlobalRandomStream
from validate_full_forward_backward import snapshot_rng,restore_rng
from validate_fresh_process_lifecycle import manifest
from activation_trace import ActivationAudit,byte_view
from compare_released_profiles import compare_state


def worker(a):
    torch.set_num_threads(2);torch.manual_seed(727);np.random.seed(727);random.seed(727)
    method=Method({'path':{'name':'flow'}});original=build('model',{'name':'tiny_conv','args':{'channels':3,'width':8}});port=copy.deepcopy(original)
    trainer=Trainer(method,port,lr=1e-4,ema_decay=.99,precision='float16',grad_clip=1.,randomness='native_global',nonfinite_policy='grad_scaler',ema_on_skipped_step=a.ema_on_skip)
    scaler=torch.amp.GradScaler('cpu',init_scale=128.);trainer.scaler=torch.amp.GradScaler('cpu',init_scale=128.)
    opt=torch.optim.AdamW(original.parameters(),lr=1e-4,weight_decay=0.);ema=copy.deepcopy(original).eval().requires_grad_(False);step=0
    generator=torch.Generator().manual_seed(771);images=torch.randn(4,3,8,8,generator=generator);noise=torch.randn(4,3,8,8,generator=generator);times=torch.rand(4,generator=generator)
    if a.resume:
        state=torch.load(a.resume/'original.pt',weights_only=False)
        original.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);scaler.load_state_dict(state['scaler']);step=state['step'];trainer.load(a.resume/'unified.pt');restore_rng(state['rng'])
    audit=ActivationAudit('cpu_scaler',[('network',original,port)],a.output/'layers');rows=[]
    for _ in range(a.steps):
        audit.begin(step);before=snapshot_rng();handles=[]
        if step==1:
            # A deterministic injected overflow tests skip semantics independently
            # of a device-specific numerical accident.
            for net in (original,port):handles.append(next(net.parameters()).register_hook(lambda grad:torch.full_like(grad,float('inf'))))
        audit.side('native');original.train();opt.zero_grad(set_to_none=True)
        with torch.autocast('cpu',dtype=torch.float16):loss=method.training(original,images,times,GlobalRandomStream(),noise=noise).loss.mean()
        scaler.scale(loss).backward();scaler.unscale_(opt);raw={k:None if p.grad is None else p.grad.detach().clone() for k,p in original.named_parameters()}
        torch.nn.utils.clip_grad_norm_(original.parameters(),1.,error_if_nonfinite=False);old_scale=scaler.get_scale();scaler.step(opt);scaler.update();skipped=scaler.get_scale()<old_scale
        with torch.no_grad():
            if a.ema_on_skip or not skipped:
                for q,p in zip(ema.parameters(),original.parameters()):q.mul_(.99).add_(p,alpha=1-.99)
                for q,p in zip(ema.buffers(),original.buffers()):q.copy_(p)
        audit.side('unified');restore_rng(before)
        def gate(net):
            for k,p in net.named_parameters():
                expected=raw[k]
                if (expected is None)!=(p.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(p.grad))):raise AssertionError('Unscaled raw gradient '+k)
        trainer.step(images,times,noise=noise,before_gradient_processing=gate)
        audit.finish_segment()
        for handle in handles:handle.remove()
        checks={k:compare_state(x.state_dict(),y.state_dict()) for k,x,y in [('model',original,port),('ema',ema,trainer.ema),('optimizer',opt,trainer.optimizer)]}
        if any(v['exact']!=v['tensors'] for v in checks.values()) or manifest(scaler.state_dict())!=manifest(trainer.scaler.state_dict()) or skipped!=trainer.last_update_skipped:raise AssertionError('Scaler/skip lifecycle differs')
        if skipped!=(step==1):raise AssertionError('Unexpected overflow pattern')
        rows.append(dict(step=step,skipped=skipped,scale=scaler.get_scale(),unscaled_raw_gradient_byte_exact=True,state=checks));step+=1
    act=audit.close()
    if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('CPU AMP layer/activation gradients differ')
    state=dict(model=original.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),scaler=scaler.state_dict(),step=step,rng=snapshot_rng())
    a.output.mkdir(exist_ok=True,parents=True);torch.save(state,a.output/'original.pt');trainer.save(a.output/'unified.pt')
    (a.output/'result.json').write_text(json.dumps(dict(status='passed',steps=rows,activation=act,original_state=manifest(state),unified_state=manifest(trainer.state_dict())),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--ema-on-skip',action='store_true');p.add_argument('--steps',type=int,default=3);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.worker:worker(a);return 0
    started=time.time();a.output.mkdir(exist_ok=True,parents=True);rows=[]
    for update in (False,True):
        case=a.output/f'ema_on_skip_{update}';case.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='umdiff_scaler_resume_') as tmp:
            tmp=Path(tmp)
            def run(mode,steps,resume=None):
                cmd=[sys.executable,__file__,'--worker','--steps',str(steps),'--output',str(tmp/mode)]
                if update:cmd+=['--ema-on-skip']
                if resume:cmd+=['--resume',str(resume)]
                with (case/(mode+'.log')).open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=90)
                if proc.returncode:raise RuntimeError(mode+' scaler check failed')
                report=json.loads((tmp/mode/'result.json').read_text());(case/(mode+'.json')).write_text(json.dumps(report,indent=2)+'\n');return report
            try:
                full=run('full',3);run('first',2);resumed=run('resumed',1,tmp/'first')
                if full['original_state']!=resumed['original_state'] or full['unified_state']!=resumed['unified_state']:raise AssertionError('Fresh-process scaler recovery differs')
                rows.append(dict(status='passed',ema_on_skipped_step=update,overflow_skip_exact=True,fresh_process_resume_after_overflow_exact=True))
            except Exception:rows.append(dict(status='failed',ema_on_skipped_step=update,failure=traceback.format_exc()))
            print(json.dumps(rows[-1]),flush=True)
    report=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',cases=rows,gpu_used=False,formal_training=False,seconds=time.time()-started,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),temporary_payloads_deleted=True,
        scope='Actual PyTorch CPU FP16 autocast/GradScaler independent optimizer lifecycle versus Trainer;controlled shared diffusion objective to isolate lifecycle,finite/forced-inf/recovered finite updates,both EMA skip policies,exact post-overflow fresh-process resume. Does not certify any paper CUDA AMP/Accelerate/FSDP loop.')
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
