#!/usr/bin/env python3
"""Verify the exact two-log-guard patch needed for continuous DFM RNG alignment."""
import ast,hashlib,json,os,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import numpy as np
import torch
from umdiff import build
from umdiff.engine import Method
from umdiff.native_models import activate
from umdiff.randomness import GlobalRandomStream
from validate_cli import equal
from validate_training import nonzero_output

def rng():return torch.get_rng_state(),np.random.get_state()
def restore(value):torch.set_rng_state(value[0]);np.random.set_state(value[1])

def main():
    torch.set_num_threads(2);activate('dfm',['models','dfm_utils','loss']);import loss as native
    path=ROOT/'sources/dfm/loss.py';raw=path.read_bytes();tree=ast.parse(raw);changed=[]
    for node in ast.walk(tree):
        if isinstance(node,ast.If) and node.lineno in (202,233):node.test=ast.Constant(value=False);changed.append(node.lineno)
    if sorted(changed)!=[202,233]:raise AssertionError('Pinned logging guards changed upstream')
    ast.fix_missing_locations(tree);namespace={'__name__':'native_dfm_logging_patch','__file__':str(path)}
    exec(compile(tree,str(path),'exec'),namespace)
    torch.manual_seed(17);np.random.seed(17)
    spec={'name':'adapter','args':{'output_index':0,'model':{'name':'native_dfm','args':{'input_size':16,'in_channels':4,'hidden_size':32,'decoder_hidden_size':32,'encoder_depth':1,'depth':2,'num_heads':4,'class_dropout_prob':0.,'enable_repa':False,'qk_norm':False}}}}
    base=nonzero_output(build('model',spec));images=torch.randn(3,4,16,16);labels=torch.tensor([0,1,2])
    method=Method({'name':'dfm_rng','batch':{'name':'dfm'}});stream=GlobalRandomStream()
    def source(objective,model):return objective(lambda x,t,**kw:(model(x,t,**kw),None),images,model_kwargs={'y':labels})[0].mean()
    def port(model):
        stage=int(np.random.choice(2,p=[.9,.1]));times=[]
        for i in range(stage+1):times.append(torch.sigmoid(torch.randn(3,1,1,1)-2) if i<stage else torch.rand(3,1,1,1))
        times.extend([torch.ones(3,1,1,1) for _ in range(1-stage)])
        return method.training(model,images,torch.stack([v.flatten() for v in times],-1),stream,model_kwargs={'y':labels},context={'stage':stage}).loss.mean()
    start=rng();source(native.DFMSILoss(stage_weights=[.9,.1]),deepcopy(base));native_end=rng()
    restore(start);port(deepcopy(base));port_end=rng()
    if equal(native_end[1],port_end[1]):raise AssertionError('Expected unpatched logging RNG mismatch was absent')
    a,b=deepcopy(base),deepcopy(base);oa=torch.optim.AdamW(a.parameters(),lr=1e-4);ob=torch.optim.AdamW(b.parameters(),lr=1e-4)
    objective=namespace['DFMSILoss'](stage_weights=[.9,.1]);state_a=state_b=start;maximum_loss=0.
    for step in range(32):
        restore(state_a);oa.zero_grad(set_to_none=True);la=source(objective,a);la.backward();oa.step();state_a=rng()
        restore(state_b);ob.zero_grad(set_to_none=True);lb=port(b);lb.backward();ob.step();state_b=rng()
        maximum_loss=max(maximum_loss,abs(la.item()-lb.item()))
        if not equal(state_a,state_b):raise AssertionError(f'Continuous RNG differs at step{step}')
        if not equal(a.state_dict(),b.state_dict()):raise AssertionError(f'Continuous parameters differ at step{step}')
        if not equal(oa.state_dict(),ob.state_dict()):raise AssertionError(f'Continuous optimizer state differs at step{step}')
    result=dict(status='passed',device='cpu',gpu_used=False,continuous_updates=32,per_step_reseeding=False,unpatched_numpy_rng_mismatch_observed=True,minimal_oracle_patch={'source':'sources/dfm/loss.py','sha256':hashlib.sha256(raw).hexdigest(),'changed_if_tests_to_false':sorted(changed),'source_file_edited':False},coupling='Port uses source NumPy stage sampling and same global Torch time/noise stream, not the generic independent Torch stage sampler.',continuous_torch_numpy_rng_bitwise=True,all_model_parameters_bitwise=True,full_adamw_state_bitwise=True,maximum_scalar_loss_abs=maximum_loss)
    (ROOT/'evidence/dfm_rng_alignment.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
