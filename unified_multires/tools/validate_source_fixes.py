#!/usr/bin/env python3
"""Reproduce source defects, verify explicitly selected compatibility paths."""
import argparse,copy,json,os,sys,time,traceback,hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import torch
from umdiff import build,native
from umdiff.native_models import activate
from umdiff.randomness import GlobalRandomStream
from activation_trace import ActivationAudit,byte_view
from validate_training import nonzero_output


@torch.no_grad()
def lapflow(out):
    activate('lapflow',['models','sampler'])
    from sampler import ours_sample as author
    from umdiff.samplers import LapFlowSampler
    spec={'name':'adapter','args':{'time_first':True,'model':{'name':'native_lapflow','args':dict(
        input_size=[16,8],patch_size=2,in_channels=4,hidden_size=32,depth=2,num_heads=4,
        class_dropout_prob=.1,use_mot=True,use_mask=True)}}}
    a=nonzero_output(build('model',spec)).eval();b=copy.deepcopy(a)
    audit=ActivationAudit('lapflow_square',[('network',a.model,b.model)],out/'layers');rows=[]
    original_noise_fn=author.square_get_noise_at_seg
    noise=torch.randn(2,4,16,16)
    defective=original_noise_fn(noise,.4)
    corrected=original_noise_fn(torch.tensor(.4),noise)
    if torch.equal(byte_view(defective),byte_view(corrected)):raise AssertionError('Negative control failed')
    old_tensor=torch.tensor
    def cpu_tensor(*args,**kwargs):
        if kwargs.get('device')=='cuda':kwargs['device']='cpu'
        return old_tensor(*args,**kwargs)
    for i,(policy,cfg,curve) in enumerate([(p,c,'square') for p in ('source_bug','corrected') for c in (1.,2.5)]+[(None,c,curve) for curve in ('linear','cosine') for c in (1.,2.5)]):
        labels=torch.arange(2);initial=torch.randn(2,4,16,16)
        both=torch.cat([initial,initial]) if cfg>1 else initial
        y=torch.cat([labels,torch.full_like(labels,1000)]) if cfg>1 else labels
        settings=SimpleNamespace(noise_var_factor=None,path_type=curve,use_gvp=curve=='cosine',epsilon=None,
            variance_aware=False,use_mot=True,num_levels=2,method='euler',step_size=.1,perturb=False,
            compute_nfe=False,cfg_scale=cfg,time_seg_small=.4,time_segments=[.3,.6],atol=1e-5,rtol=1e-5)
        audit.begin(i);audit.side('native')
        fn=original_noise_fn if policy=='source_bug' else lambda n,t:original_noise_fn(old_tensor(t,device=n.device,dtype=n.dtype),n)
        with patch.object(author.torch,'tensor',cpu_tensor),patch.object(author,'square_get_noise_at_seg',fn):
            expected=author.ours_sample_from_model([a.model],0,both,{'y':y,**({'cfg_scale':cfg} if cfg>1 else {})},settings,False,largest_only=True)[:2]
        audit.side('unified')
        actual=LapFlowSampler(curve=curve,square_policy=policy or 'reject',guidance_scale=cfg)(b,initial,GlobalRandomStream(),{'y':labels})
        if not torch.equal(byte_view(actual),byte_view(expected)):raise AssertionError(f'{curve}/{policy}/CFG{cfg}: max_abs={float((actual-expected).abs().max())}')
        audit.finish_segment();rows.append(dict(policy=policy,cfg=cfg,curve=curve,float_pixels_byte_exact=True))
    return dict(checks=rows,activation=audit.close(),negative_control=dict(
        original_swapped_call_is_wrong=True,max_abs=float((defective-corrected).abs().max())),
        scope='Explicit source_bug parity and explicit corrected noise/time argument order + tensor scalar promotion; original files untouched. Reduced random author backbone,CPU FP32.')


def ppflow_case(out,levels):
    from validate_paired_experiments import reduced
    from umdiff.source_objectives import setup
    c=json.loads((ROOT/f'configs/paired_experiments/ppflow_xl_{levels}.json').read_text());c,_=reduced(c)
    c['model']['args']['learn_sigma']=False
    images=torch.randn(8,*c['image_shape']);labels=torch.arange(8)
    _,original,port,_,_=setup('ppflow' if levels==2 else 'ppflow3',c,images,labels)
    bad=nonzero_output(build('model',c['model']));failed=None
    try:original(bad,0)
    except (RuntimeError,AssertionError) as exc:failed=str(exc)
    if failed is None:raise AssertionError('Expected source learn_sigma=False unpatchify defect')
    good_spec=copy.deepcopy(c['model']);good_spec['name']='native_ppflow_no_sigma_compat'
    a=nonzero_output(build('model',good_spec));b=copy.deepcopy(a)
    audit=ActivationAudit(f'ppflow{levels}_no_sigma',[('network',a,b)],out/'layers');rows=[]
    for i in range(2):
        a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
        audit.begin(i);audit.side('native');torch.manual_seed(911+i);la=original(a,i).mean();la.backward()
        audit.side('unified');torch.manual_seed(911+i);lb=port(b,i).mean();lb.backward()
        audit.parameters(a,b);audit.finish_segment()
        if not torch.equal(byte_view(la),byte_view(lb)):raise AssertionError('Loss bytes differ')
        rows.append(dict(step=i,loss_byte_exact=True))
    from umdiff.samplers import PPFlowNativeSampler
    from transport.transport import Transport,ModelType,PathType,WeightType,Sampler
    initial=torch.randn(2,*c['image_shape']);y=torch.arange(2);a.eval();b.eval()
    objective=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,loss_type=WeightType.NONE,train_eps=0.,sample_eps=0.)
    for cfg in (None,[2.5]*levels):
        with torch.no_grad():
            audit.begin(2 if cfg is None else 3);audit.side('native')
            x=initial if cfg is None else torch.cat([initial,initial]);kw={'y':y if cfg is None else torch.cat([y,torch.full_like(y,1000)])}
            if cfg is not None:kw['cfg_scales']=cfg
            expected=Sampler(objective).sample_ode(sampling_method='heun2',num_steps=13)(x,a.forward_wo_cfg if cfg is None else a.forward_with_cfg,**kw)[-1][:2]
            audit.side('unified');actual=PPFlowNativeSampler(points=13,cfg_scales=cfg)(b,initial,GlobalRandomStream(),model_kwargs={'y':y})
            if not torch.equal(byte_view(actual),byte_view(expected)):raise AssertionError('No-sigma inference pixels differ')
            audit.finish_segment();rows.append(dict(cfg=cfg,heun2_float_pixels_byte_exact=True))
    return dict(levels=levels,checks=rows,activation=audit.close(),negative_control=dict(author_false_branch_failed=failed),
        compatibility=a.compatibility_evidence,scope='Published no-sigma defect reproduced;two-expression opt-in correction tested with independent objectives and native Heun2 inference with/withoutCFG;random reduced architecture,CPU FP32.')


def ppflow(out):
    cases=[ppflow_case(out,n) for n in (2,3)]
    return dict(cases=cases,activation=cases[0]['activation'],additional_activation=cases[1]['activation'])


RUNNERS={'lapflow':lapflow,'ppflow':ppflow}
def main():
    p=argparse.ArgumentParser();p.add_argument('method',choices=RUNNERS);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.manual_seed(20261012);started=time.time()
    report=dict(method=a.method,gpu_used=False,formal_training=False,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        report.update(RUNNERS[a.method](a.output))
        for act in [report['activation']]+([report['additional_activation']] if 'additional_activation' in report else []):
            assert not act['structural_mismatches'] and all(v['pairs']==v['byte_exact'] for v in act['totals'].values())
        report['status']='passed'
    except Exception:report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
    report.update(seconds=time.time()-started,source_executions=native.AUDIT,cuda_initialized=torch.cuda.is_initialized())
    (a.output/(a.method+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('method','status','seconds')}),flush=True)
    return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
