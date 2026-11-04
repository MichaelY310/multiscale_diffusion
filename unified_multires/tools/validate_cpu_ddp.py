#!/usr/bin/env python3
"""Real two-rank CPU DDP parity, synchronization policy and fresh resume."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,tempfile,time,traceback
from pathlib import Path
from contextlib import nullcontext
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build
from umdiff.engine import Method
from umdiff.source_objectives import setup
from umdiff.training import Trainer
from validate_training import nonzero_output
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from validate_fresh_process_lifecycle import manifest
from activation_trace import ActivationAudit,byte_view
from compare_released_profiles import compare_state


def worker(a):
    torch.set_num_threads(2);torch.manual_seed(644+a.rank);np.random.seed(644+a.rank);random.seed(644+a.rank)
    torch.distributed.init_process_group('gloo',init_method=a.rendezvous,rank=a.rank,world_size=2)
    c=json.loads((ROOT/'configs/native_fixtures/ppflow_2.json').read_text());c['model']['args']['class_dropout_prob']=.1
    base=nonzero_output(build('model',c['model']));left=torch.nn.parallel.DistributedDataParallel(base)
    right=torch.nn.parallel.DistributedDataParallel(copy.deepcopy(left.module));ema=copy.deepcopy(left.module).eval().requires_grad_(False)
    trainer=Trainer(Method(c),right,lr=1e-4,ema_decay=.9999,accumulation_steps=2,optimizer_parameters='all',randomness='native_global',distributed_sync=a.policy)
    opt=torch.optim.AdamW(left.parameters(),lr=1e-4,weight_decay=0.);step=0
    if a.resume:
        source=a.resume/f'rank{a.rank}';state=torch.load(source/'original.pt',weights_only=False)
        left.module.load_state_dict(state['model']);ema.load_state_dict(state['ema']);opt.load_state_dict(state['optimizer']);step=state['step'];trainer.load(source/'unified.pt');restore_rng(state['rng'])
    generator=torch.Generator().manual_seed(900+a.rank);images=torch.randn(16,4,16,16,generator=generator);labels=torch.arange(16)+a.rank*16
    out=a.output/f'rank{a.rank}';out.mkdir(parents=True,exist_ok=True);audit=ActivationAudit('ppflow_ddp',[('network',left.module,right.module)],out/'layers');rows=[]
    for _ in range(a.steps):
        audit.begin(step);audit.side('native');before=snapshot_rng();left.train();opt.zero_grad(set_to_none=True)
        for i in range(2):
            scope=left.no_sync() if a.policy=='last_microbatch' and i==0 else nullcontext()
            with scope:
                _,native_loss,_,_,_=setup('ppflow',c,images[8*i:8*i+8],labels[8*i:8*i+8]);native_loss(left,step).mean().div(2).backward()
        raw={k:None if p.grad is None else p.grad.detach().clone() for k,p in left.named_parameters()};opt.step()
        with torch.no_grad():
            for q,p in zip(ema.parameters(),left.module.parameters()):q.mul_(.9999).add_(p,alpha=1-.9999)
        after=snapshot_rng();audit.side('unified');restore_rng(before)
        def gate(net):
            for key,p in net.named_parameters():
                expected=raw[key]
                if (expected is None)!=(p.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(p.grad))):raise AssertionError('All-reduced raw gradient '+key)
        trainer.step(images,None,model_kwargs={'y':labels},before_gradient_processing=gate);audit.parameters(left.module,right.module);audit.finish_segment()
        checks={k:compare_state(x.state_dict(),y.state_dict()) for k,x,y in [('model',left,right),('ema',ema,trainer.ema.module),('optimizer',opt,trainer.optimizer)]}
        if any(v['exact']!=v['tensors'] for v in checks.values()) or fingerprint(after)!=fingerprint(snapshot_rng()):raise AssertionError('DDP state/RNG differs')
        digests=[None,None];torch.distributed.all_gather_object(digests,manifest(left.module.state_dict()))
        if digests[0]!=digests[1]:raise AssertionError('Parameters differ across ranks')
        rows.append(dict(step=step,state=checks,raw_gradient_exact=True,rank_weights_exact=True));step+=1
    act=audit.close()
    if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('DDP layer/activation gradients differ')
    state=dict(model=left.module.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),step=step,rng=snapshot_rng())
    torch.save(state,out/'original.pt');trainer.save(out/'unified.pt')
    (out/'result.json').write_text(json.dumps(dict(status='passed',rank=a.rank,policy=a.policy,steps=rows,activation=act,original_state=manifest(state),unified_state=manifest(trainer.state_dict())),indent=2)+'\n')
    torch.distributed.destroy_process_group()


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--rank',type=int);p.add_argument('--rendezvous');p.add_argument('--policy',default='every_microbatch');p.add_argument('--steps',type=int,default=4);p.add_argument('--output',type=Path,required=True);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.worker:worker(a);return 0
    started=time.time();a.output.mkdir(exist_ok=True,parents=True);rows=[]
    for policy in ('every_microbatch','last_microbatch'):
        case=a.output/policy;case.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='umdiff_cpu_ddp_') as tmp:
            tmp=Path(tmp)
            def run(mode,steps,resume=None):
                jobs=[];logs=[]
                for rank in (0,1):
                    cmd=[sys.executable,__file__,'--worker','--rank',str(rank),'--rendezvous','file://'+str(tmp/('rdzv_'+mode)),'--policy',policy,'--steps',str(steps),'--output',str(tmp/mode)]
                    if resume:cmd+=['--resume',str(resume)]
                    log=(case/f'{mode}_rank{rank}.log').open('w');logs.append(log);jobs.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT))
                try:
                    for job in jobs:
                        if job.wait(timeout=180):raise RuntimeError(mode+' CPU DDP worker failed')
                finally:
                    for job in jobs:
                        if job.poll() is None:job.kill();job.wait()
                    for log in logs:log.close()
                results=[]
                for rank in (0,1):
                    result=json.loads((tmp/mode/f'rank{rank}/result.json').read_text());(case/f'{mode}_rank{rank}.json').write_text(json.dumps(result,indent=2)+'\n');results.append(result)
                return results
            try:
                full=run('full',4);run('first',2);resumed=run('resumed',2,tmp/'first')
                for x,y in zip(full,resumed):
                    if x['original_state']!=y['original_state'] or x['unified_state']!=y['unified_state']:raise AssertionError('Two-rank fresh-process resume differs')
                rows.append(dict(policy=policy,status='passed',ranks=2,fresh_process_exact=True,per_layer_and_raw_allreduced_gradient_exact=True))
            except Exception:rows.append(dict(policy=policy,status='failed',failure=traceback.format_exc()))
            print(json.dumps(rows[-1]),flush=True)
    report=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',cases=rows,gpu_used=False,formal_training=False,seconds=time.time()-started,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        temporary_payloads_deleted=True,scope='Real two-rank Gloo DDP,rank-distinct synthetic latents,actual author PPFlow transport/backward versus independent unified loss+Trainer;every/last microbatch allreduce and fresh-process resume. Reduced random network;NCCL/FSDP/CUDA and original best-run worldsize not certified.')
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
