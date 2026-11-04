#!/usr/bin/env python3
"""Published codec backbone/loss + independent optimizer, fresh-process resume."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff.training import Trainer
from umdiff.engine import TrainingResult
from validate_training import nonzero_output
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from validate_fresh_process_lifecycle import manifest
from activation_trace import ActivationAudit,byte_view
from compare_released_profiles import compare_state
import validate_training_more as subject
import conformance
CURRENT_AUDIT=None


def worker(a):
    global CURRENT_AUDIT
    torch.set_num_threads(2);torch.manual_seed(481);np.random.seed(481);random.seed(481);reports=[]
    def verify(name,model,original,port,steps,**unused):
        global CURRENT_AUDIT
        first=copy.deepcopy(model);second=copy.deepcopy(model);ema=copy.deepcopy(first).eval().requires_grad_(False)
        opt=torch.optim.AdamW(first.parameters(),lr=1e-4,weight_decay=0.)
        class Port:
            config={'codec_component':name,'validation_only':True}
            def training(self,net,x,t,stream,noise=None,model_kwargs=None,active=None,context=None):return TrainingResult(port(net,context['step']).mean(),None,None,None,{})
        trainer=Trainer(Port(),second,lr=1e-4,ema_decay=.9999,optimizer_parameters='all',randomness='native_global');step=0
        out=a.output/name.replace('/','_');out.mkdir(parents=True,exist_ok=True)
        torch.manual_seed(482+len(reports));np.random.seed(482+len(reports));random.seed(482+len(reports))
        if a.resume:
            source=a.resume/name.replace('/','_');saved=torch.load(source/'original.pt',weights_only=False)
            first.load_state_dict(saved['model']);ema.load_state_dict(saved['ema']);opt.load_state_dict(saved['optimizer']);step=saved['step'];trainer.load(source/'unified.pt');restore_rng(saved['rng'])
        audit=ActivationAudit(name,[('codec',first.model,second.model)],out/'layers');CURRENT_AUDIT=audit;rows=[]
        for _ in range(steps):
            audit.begin(step);audit.side('native');before=snapshot_rng();first.train();opt.zero_grad(set_to_none=True);loss=original(first,step).mean();loss.backward()
            raw={k:None if p.grad is None else p.grad.detach().clone() for k,p in first.named_parameters()};opt.step()
            with torch.no_grad():
                for q,p in zip(ema.parameters(),first.parameters()):q.mul_(.9999).add_(p,alpha=1-.9999)
                for q,p in zip(ema.buffers(),first.buffers()):q.copy_(p)
            after=snapshot_rng();audit.side('unified');restore_rng(before)
            def gate(net):
                for k,p in net.named_parameters():
                    expected=raw[k]
                    if (expected is None)!=(p.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(p.grad))):raise AssertionError('Raw codec gradient '+k)
            result=trainer.step(torch.empty(2,3,16,16),None,before_gradient_processing=gate)
            audit.parameters(first.model,second.model,'codec');audit.finish_segment()
            checks={key:compare_state(x.state_dict(),y.state_dict()) for key,x,y in [('model',first,second),('ema',ema,trainer.ema),('optimizer',opt,trainer.optimizer)]}
            if any(v['exact']!=v['tensors'] for v in checks.values()) or fingerprint(after)!=fingerprint(snapshot_rng()) or not torch.equal(byte_view(loss),byte_view(result.loss)):raise AssertionError('Codec state/loss/RNG mismatch')
            rows.append(dict(step=step,state=checks,raw_gradient_exact=True,rng_exact=True));step+=1
        act=audit.close();CURRENT_AUDIT=None
        if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('Codec layer/activation gradient mismatch')
        state=dict(model=first.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,rng=snapshot_rng());torch.save(state,out/'original.pt');trainer.save(out/'unified.pt')
        reports.append(dict(variant=name,steps=rows,activation=act,original_state=manifest(state),unified_state=manifest(trainer.state_dict())))
        return first,second,{'status':'passed'}
    subject.install(verify,nonzero_output,conformance.compare);details=subject.run_diffusability(a.steps)
    report=dict(status='passed',variants=reports,details=details,scope='Actual released codec backbone/reconstruction/SE methods, independent AdamW/EMA vsTrainer;mean/bilinear SE2,random reduced real AutoencoderKL,CPU FP32. Native codec full Hydra/pretrainedVGG/GAN/temporal best recipe remains separate.')
    (a.output/'result.json').write_text(json.dumps(report,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--steps',type=int,default=4);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.worker:worker(a);return 0
    started=time.time();a.output.mkdir(exist_ok=True,parents=True);report=dict(gpu_used=False,formal_training=False)
    with tempfile.TemporaryDirectory(prefix='umdiff_codec_resume_') as tmp:
        tmp=Path(tmp)
        def run(mode,steps,resume=None):
            cmd=[sys.executable,__file__,'--worker','--steps',str(steps),'--output',str(tmp/mode)]
            if resume:cmd+=['--resume',str(resume)]
            with (a.output/(mode+'.log')).open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode:raise RuntimeError(mode+' failed')
            value=json.loads((tmp/mode/'result.json').read_text());(a.output/(mode+'.json')).write_text(json.dumps(value,indent=2)+'\n');return value
        try:
            full=run('full',4);run('first',2);resumed=run('resumed',2,tmp/'first')
            for x,y in zip(full['variants'],resumed['variants']):
                if x['original_state']!=y['original_state'] or x['unified_state']!=y['unified_state']:raise AssertionError('Fresh-process codec state differs')
            report.update(status='passed',variants=[v['variant'] for v in full['variants']],original_fresh_process_exact=True,unified_fresh_process_exact=True)
        except Exception:report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
    report.update(seconds=time.time()-started,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),temporary_payloads_deleted=True)
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True);return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
