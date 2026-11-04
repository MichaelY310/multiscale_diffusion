#!/usr/bin/env python3
"""Independent native phase loops versus RMDTrainer, including fresh resume."""
import argparse,copy,hashlib,json,os,random,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from umdiff.distillation import RMDTrainer
from umdiff import native
from validate_rmd import fixtures
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from validate_fresh_process_lifecycle import manifest
from activation_trace import ActivationAudit,byte_view
from compare_released_profiles import compare_state


def worker(a):
    torch.set_num_threads(2);torch.manual_seed(852);np.random.seed(852);random.seed(852)
    src,loss,env,cfg,p,obj,base,template,text,clock,high=fixtures(a.transition,warmup_step=2)
    config=dict(task='rmd',image_shape=[4,2,4,6],distillation={**vars(cfg),'target_size':[8,12]},training={'accumulation_steps':2})
    config['distillation'].pop('gan');trainer=RMDTrainer(config,copy.deepcopy(base),p.uncond_prompt_embeds)
    model=copy.deepcopy(base);fake=copy.deepcopy(base);real=copy.deepcopy(base).eval().requires_grad_(False)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-6,betas=(0.,.95),weight_decay=1e-4);optf=torch.optim.AdamW(fake.parameters(),lr=2e-5,betas=(0.,.95),weight_decay=1e-4)
    step=0;template=template.repeat(2,1,1,1,1);text=text.repeat(2,1,1)
    if a.resume:
        state=torch.load(a.resume,weights_only=False)
        for key,value in [('model',model),('fake',fake),('real',real),('optimizer',opt),('fake_optimizer',optf)]:value.load_state_dict(state[key])
        step=state['step'];trainer.load(a.resume.with_name('unified.pt'));restore_rng(state['rng'])
    audit=ActivationAudit('rmd_transition'+str(a.transition),[('generator',model,trainer.model),('fake',fake,trainer.fake),('teacher',real,trainer.real)],a.output/'layers');rows=[]
    for _ in range(a.steps):
        audit.begin(step);audit.side('native');before=snapshot_rng();opt.zero_grad(set_to_none=True);optf.zero_grad(set_to_none=True);model.train();fake.train();raw={}
        for micro in range(2):
            inputs=template[2*micro:2*micro+2];condition=text[2*micro:2*micro+2];torch.randn(inputs.shape);noise=torch.randn_like(inputs);model.eval()
            with torch.no_grad():clean,trajectory=p.generate_new_upsample(model,None,noise,noise,condition,None,steps=6,eta=.5,return_mid=True,flow_shift_trans=a.transition,relusion_shift=2,target_size=(8,12))
            trajectory.reverse();indices=torch.randint(4,6,(2,));example=torch.randn_like(clean[int(indices[0])])
            for j in range(2):example[j]=clean[int(indices[j])][j]
            model.train()
            for phase,optimizer,network in [(1,optf,fake),(2,opt,model)]:
                args=(cfg,p,model,fake)+((real,) if phase==2 else ())+(clock,high,example,trajectory,condition,None,6,4,obj.scales,1000,torch.float32,'cpu')
                if phase==2:args+=(None,)
                args+=(step,);value=env['_phase'+str(phase)](*args,target_size=(8,12))/2;value.backward()
                if micro==1:
                    raw[phase]={k:None if q.grad is None else q.grad.detach().clone() for k,q in network.named_parameters()}
                    if phase==2:torch.nn.utils.clip_grad_norm_(network.parameters(),1.)
                    optimizer.step();optimizer.zero_grad(set_to_none=True)
        after=snapshot_rng();audit.side('unified');restore_rng(before)
        def gate(phase,network):
            for key,q in network.named_parameters():
                expected=raw[phase][key]
                if (expected is None)!=(q.grad is None) or (expected is not None and not torch.equal(byte_view(expected),byte_view(q.grad))):raise AssertionError(f'Raw gradient phase{phase}/{key}')
        trainer.step(template,text,before_gradient_processing=gate);audit.finish_segment()
        checks={k:compare_state(u.state_dict(),v.state_dict()) for k,u,v in [('model',model,trainer.model),('fake',fake,trainer.fake),('real',real,trainer.real),('optimizer',opt,trainer.optimizer),('fake_optimizer',optf,trainer.fake_optimizer)]}
        if any(v['exact']!=v['tensors'] for v in checks.values()) or fingerprint(after)!=fingerprint(snapshot_rng()):raise AssertionError('RMD lifecycle state/RNG mismatch')
        rows.append(dict(step=step,state=checks,raw_gradients_byte_exact=True,rng_byte_exact=True));step+=1
    act=audit.close()
    if act['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in act['totals'].values()):raise AssertionError('RMD layer values/activation gradients differ')
    state=dict(model=model.state_dict(),fake=fake.state_dict(),real=real.state_dict(),optimizer=opt.state_dict(),fake_optimizer=optf.state_dict(),step=step,rng=snapshot_rng())
    a.output.mkdir(exist_ok=True,parents=True);torch.save(state,a.output/'original.pt');trainer.save(a.output/'unified.pt')
    (a.output/'result.json').write_text(json.dumps(dict(status='passed',transition=a.transition,steps=rows,activation=act,
        original_state=manifest(state),unified_state=manifest(torch.load(a.output/'unified.pt',weights_only=False)),source_executions=native.AUDIT,
        scope='Actual source Predictor/phase1/phase2, independent two-optimizer interleaving and unprocessed gradients;RMDTrainer resume in a fresh process. Reduced random Wan,synthetic text/video,CPU FP32;warmup2 checked around boundary,official warmup15 remains in prior separate tests.'),indent=2)+'\n')


def main():
    p=argparse.ArgumentParser();p.add_argument('--worker',action='store_true');p.add_argument('--output',type=Path,required=True);p.add_argument('--transition',type=int,default=1);p.add_argument('--steps',type=int,default=4);p.add_argument('--resume',type=Path);a=p.parse_args()
    if a.worker:worker(a);return 0
    a.output.mkdir(exist_ok=True,parents=True);rows=[];started=time.time()
    for transition in (1,2):
        base=a.output/f'transition{transition}';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='umdiff_rmd_resume_') as tmp:
            tmp=Path(tmp)
            def run(kind,steps,resume=None):
                cmd=[sys.executable,__file__,'--worker','--transition',str(transition),'--steps',str(steps),'--output',str(tmp/kind)]
                if resume:cmd+=['--resume',str(resume)]
                with (base/(kind+'.log')).open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
                if result.returncode:raise RuntimeError(kind+' failed')
                report=json.loads((tmp/kind/'result.json').read_text());(base/(kind+'.json')).write_text(json.dumps(report,indent=2)+'\n');return report
            try:
                full=run('full',4);run('first',2);resumed=run('resumed',2,tmp/'first/original.pt')
                if full['original_state']!=resumed['original_state'] or full['unified_state']!=resumed['unified_state']:raise AssertionError('Fresh-process final state mismatch')
                rows.append(dict(transition=transition,status='passed',fresh_process_original_exact=True,fresh_process_unified_exact=True))
            except Exception:rows.append(dict(transition=transition,status='failed',failure=traceback.format_exc()))
            print(json.dumps(rows[-1]),flush=True)
    report=dict(status='passed' if all(r['status']=='passed' for r in rows) else 'failed',cases=rows,gpu_used=False,formal_training=False,seconds=time.time()-started,temporary_payloads_deleted=True,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
