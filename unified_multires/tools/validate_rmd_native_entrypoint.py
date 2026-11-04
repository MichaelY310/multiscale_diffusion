#!/usr/bin/env python3
"""Run released RMD train() itself, replacing only asset factories/geometry."""
import json,os,sys,time,traceback
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'sources/rmd/src')]
os.environ.setdefault('CUDA_VISIBLE_DEVICES','');os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
import torch
torch.set_num_threads(2)
from umdiff import build
from validate_rmd import MODEL
from rmd.config import TrainConfig
import rmd.engines.train as source

def main():
    started=time.time();directory=ROOT/'runs/rmd_native_entrypoint';directory.mkdir(parents=True,exist_ok=True)
    prompts=directory/'synthetic_prompts.txt';prompts.write_text('local synthetic fixture zero\nlocal synthetic fixture one\n')
    torch.manual_seed(7301);base=build('model',MODEL)
    source.LATENT_SIZES={480:(4,2,4,6),720:(4,2,8,12)}
    source.load_tokenizer_text_encoder=lambda *args:(None,None)
    def embeddings(tokenizer,encoder,prompts,*args,**kwargs):
        return {prompt:torch.randn(3,8,generator=torch.Generator().manual_seed(sum(map(ord,prompt)))) for prompt in set(prompts)}
    source.precompute_prompt_embeddings=embeddings
    source.load_training_transformer=lambda cfg,device=None,frozen=False:deepcopy(base).to(device=device).requires_grad_(not frozen)
    cfg=TrainConfig(device='cpu',mixed_precision='no',gradient_checkpointing=False,train_batch_size=2,max_train_steps=2,checkpointing_steps=1,output_dir=str(directory),data_path=str(prompts),k_step=6,low_rs_step=3,flow_shift_trans=1,t_offset_factor=.25,split_timestep=True,warmup_step=3,lr_warmup_steps=0,validation_prompt_path=None)
    failure=None
    try:
        source.train(cfg)
        from safetensors.torch import load_file
        result=load_file(str(directory/'student_model-2/model.safetensors'))
        if not all(torch.isfinite(x).all() for x in result.values()):raise AssertionError('Non-finite native checkpoint')
        delta=max((result[k]-v.cpu()).abs().max().item() for k,v in base.state_dict().items())
        if delta==0:raise AssertionError('Native full training did not update generator')
        if not (directory/'checkpoint-2').is_dir():raise AssertionError('Native Accelerator checkpoint missing')
    except Exception:failure=traceback.format_exc();print(failure,file=sys.stderr);delta=None
    report=dict(status='failed' if failure else 'passed',device='cpu',gpu_used=False,seconds=time.time()-started,steps=2,actual_entrypoint='sources/rmd/src/rmd/engines/train.py:train',changes=['Only loader factories supply random local Wan and synthetic text embeddings','LATENT_SIZES replaced by small BCTHW geometry','Actual Accelerator, prompt DataLoader, two optimizers, LR scheduler, backward/clip, model export and training-state checkpoint execute'],generator_weight_delta=delta,failure=failure)
    (ROOT/'evidence/rmd_native_entrypoint.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return failure is not None
if __name__=='__main__':raise SystemExit(main())
