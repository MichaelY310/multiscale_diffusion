#!/usr/bin/env python3
"""Unchanged DFM loss vs native policy, continuous RNG without reseeding."""
import json,os,sys,time
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build,native
from umdiff.engine import Method
from umdiff.native_models import activate
from umdiff.randomness import GlobalRandomStream
from activation_trace import ActivationAudit
from compare_released_profiles import compare_state,named_optimizer_state,rng_fingerprint
from validate_training import nonzero_output
from validate_cli import equal
OUT=ROOT/'evidence/current_recipe_parity_20261009/continuous_rng'
def capture():return torch.get_rng_state(),np.random.get_state()
def restore(value):torch.set_rng_state(value[0]);np.random.set_state(value[1])
def main():
    OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);activate('dfm',['models','dfm_utils','loss']);import loss as author
    torch.manual_seed(17);np.random.seed(17);started=time.time()
    spec={'name':'adapter','args':{'output_index':0,'model':{'name':'native_dfm','args':{'input_size':16,'in_channels':4,'hidden_size':32,'decoder_hidden_size':32,'encoder_depth':1,'depth':2,'num_heads':4,'class_dropout_prob':.1,'enable_repa':False,'qk_norm':False}}}}
    base=nonzero_output(build('model',spec));images=torch.randn(3,4,16,16);labels=torch.arange(3);rows=[]
    for weighting,repa in [('uniform',False),('lognormal',False),('uniform',True)]:
        if repa:
            repa_spec=deepcopy(spec['args']['model']);repa_spec['args'].update(enable_repa=True,z_dims=[8],projector_dim=32)
            a=nonzero_output(build('model',repa_spec));features=[torch.randn(3,64,8)]
        else:a=deepcopy(base);features=None
        b=deepcopy(a)
        opts=[torch.optim.AdamW(net.parameters(),lr=1e-4,weight_decay=0.) for net in [a,b]]
        method=Method({'batch':{'name':'dfm','args':{'weighting':weighting,'native_debug_rng':True,'repa_weight':.3 if repa else 0.}}})
        clock=build('time_sampler',{'name':'dfm','args':{'weighting':weighting}});objective=author.DFMSILoss(stage_weights=[.9,.1],weighting=weighting)
        audit=ActivationAudit('dfm_continuous_'+weighting+('_repa' if repa else ''),[('network',a,b)],OUT)
        state_a=state_b=capture();loss_max=0.
        for step in range(128):
            audit.begin(step);audit.side('native');restore(state_a);opts[0].zero_grad(set_to_none=True)
            denoising,projection=objective(a if repa else lambda x,t,**kw:(a(x,t,**kw),None),images,model_kwargs={'y':labels},zs=features)
            native_loss=(denoising+(.3*projection if repa else 0.)).mean()
            audit.side('native','backward_recompute');native_loss.backward();torch.nn.utils.clip_grad_norm_(a.parameters(),1.);opts[0].step();state_a=capture()
            audit.side('unified');restore(state_b);opts[1].zero_grad(set_to_none=True)
            t,context=clock(len(images),'cpu',GlobalRandomStream())
            if repa:context['teacher_features']=features
            port_loss=method.training(b,images,t,GlobalRandomStream(),model_kwargs={'y':labels},context=context).loss.mean()
            audit.side('unified','backward_recompute');port_loss.backward();torch.nn.utils.clip_grad_norm_(b.parameters(),1.);opts[1].step();state_b=capture()
            audit.parameters(a,b);audit.finish_segment()
            loss_max=max(loss_max,abs(native_loss.item()-port_loss.item()))
            if not equal(state_a,state_b):raise AssertionError(f'Continuous RNG differs at {weighting}/{step}')
            if not equal(a.state_dict(),b.state_dict()):raise AssertionError(f'Continuous model differs at {weighting}/{step}')
            if not equal(opts[0].state_dict(),opts[1].state_dict()):raise AssertionError(f'Continuous optimizer differs at {weighting}/{step}')
        activation=audit.close()
        if activation['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in activation['totals'].values()):raise AssertionError('Continuous activation differs')
        rows.append(dict(weighting=weighting,repa=repa,teacher_features='synthetic local tensors' if repa else None,steps=128,per_step_reseeding=False,author_logging_guards_patched=False,continuous_torch_numpy_rng_byte_exact=True,
             activation=activation,loss_scalar_max_abs=loss_max,optimizer=compare_state(named_optimizer_state(opts[0],a),named_optimizer_state(opts[1],b))))
    report=dict(status='passed',device='cpu',gpu_used=False,seconds=time.time()-started,variants=rows,source_executions=native.AUDIT,
       scope='Reduced nonzero author networks, synthetic input, original unpatched logging RNG, source optimizer/clip; no quality claim.')
    (OUT/'dfm.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ['status','seconds']}))
if __name__=='__main__':main()
