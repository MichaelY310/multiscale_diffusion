#!/usr/bin/env python3
"""Strict CAT supplement codec forward, KL and diagnostic-adjoint parity."""
import argparse, ast, copy, hashlib, json, os, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from activation_trace import ActivationAudit,byte_view
from umdiff.compat.cat_codec import load_author,forward_components

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);p.add_argument('--steps',type=int,default=2);p.add_argument('--full-backbone',action='store_true');a=p.parse_args()
    src=ROOT/'evidence/forward_backward_20261012/cat_release/code/CAT Supplementary Material/code'
    author=load_author(src/'model');torch.set_num_threads(a.threads);torch.manual_seed(20261012)
    # The only published explicit architecture config is in classification.py.
    tree=ast.parse((src/'downstream tasks/classification.py').read_text())
    config=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='vae_config' for t in n.targets)))
    if not a.full_backbone:config.update(sample_size=64,block_out_channels=[32]*6,layers_per_block=1,num_layers=1)
    resolution=config['sample_size']
    native=author(**config);unified=copy.deepcopy(native)
    opt_a=torch.optim.SGD(native.parameters(),lr=1e-5);opt_b=torch.optim.SGD(unified.parameters(),lr=1e-5)
    a.output.mkdir(parents=True,exist_ok=True)
    # The composed port deliberately bypasses the author's root forward.
    audit=ActivationAudit('cat_published_codec',[('network',native,unified)],a.output,excluded_modules=('',))
    rows=[];started=time.time()
    for step in range(a.steps):
      for rates in ((1,1,1),(2,2,2),(3,3,3),(1,2,3)):
       for posterior_sampling in (False,True):
        images=torch.randn(3,3,resolution,resolution);r=torch.tensor(rates,dtype=torch.long)
        opt_a.zero_grad(set_to_none=True);opt_b.zero_grad(set_to_none=True)
        audit.begin(step,f'rates{rates}/sample{posterior_sampling}');audit.side('native')
        before=torch.get_rng_state();pixels_a,kl_a=native(images,r,sample_posterior=posterior_sampling);after=torch.get_rng_state()
        audit.side('unified');torch.set_rng_state(before);pixels_b,kl_b=forward_components(unified,images,r,posterior_sampling)
        assert torch.equal(after,torch.get_rng_state())
        assert pixels_a.dtype==pixels_b.dtype and torch.equal(byte_view(pixels_a),byte_view(pixels_b))
        assert torch.equal(byte_view(kl_a),byte_view(kl_b))
        # No author trainer is released. This specified adjoint probe exercises
        # both encoder and decoder; it is NOT a claimed author training loss.
        loss_a=pixels_a.square().mean()+kl_a;loss_b=pixels_b.square().mean()+kl_b
        assert torch.equal(byte_view(loss_a),byte_view(loss_b))
        audit.side('native','backward_recompute');loss_a.backward()
        audit.side('unified','backward_recompute');loss_b.backward()
        audit.parameters(native,unified);audit.finish_segment();opt_a.step();opt_b.step()
        for pa,pb in zip(native.parameters(),unified.parameters()):assert torch.equal(byte_view(pa),byte_view(pb))
        rows.append(dict(step=step,rates=rates,sample_posterior=posterior_sampling,output_shape=list(pixels_a.shape),kl=float(kl_a.detach()),diagnostic_loss=float(loss_a.detach())))
    result=audit.close();assert not result['structural_mismatches']
    assert all(s['pairs']>0 and s['pairs']==s['byte_exact'] for s in result['totals'].values())
    report=dict(method='cat',status='passed',device='cpu',gpu_used=False,steps=a.steps,seconds=time.time()-started,parameters=sum(x.numel() for x in native.parameters()),configuration=config,full_released_architecture=a.full_backbone,activation=result,cases=rows,
      source_sha256={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(src.rglob('*.py'))},harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((ROOT/'umdiff/compat/cat_codec.py').read_bytes()).hexdigest(),
      official_training_loss_verified=False,scope='Published nested-VAE forward/KL assembly, all compression1/2/3 plus mixed rates, posterior mean/sample. '+('Full classification.py architecture and512-grid. ' if a.full_backbone else 'Reduced width/depth/64-grid, author six-block topology. ')+'Diagnostic pixel.square().mean()+source KL adjoint, not a released author trainer or diffusion experiment.',
      release_gaps=['Supplement contains two codec files, scoring and classification scripts, no diffusion trainer/full codec recipe or model weights.','Author homogeneous-rate decoder produces16/32/64 pixels for rates1/2/3 on64-grid; behavior preserved, not silently resized.','Tiling/slicing and optional replacement Megatron attention have missing/incompatible call dependencies; not enabled or certified.','No LLM scoring/assets/private Shutterstock data loaded.'])
    (a.output/'cat_codec.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('method','status','seconds')}),flush=True)
if __name__=='__main__':main()
