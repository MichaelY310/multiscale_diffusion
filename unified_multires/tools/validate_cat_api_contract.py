#!/usr/bin/env python3
"""Published CAT root API and explicit-generator diagnostic adjoint parity."""
import argparse,ast,copy,hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from activation_trace import ActivationAudit,byte_view
from umdiff.compat.cat_codec import CATCodecAdapter,load_author

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    src=ROOT/'evidence/forward_backward_20261012/cat_release/code/CAT Supplementary Material/code'
    tree=ast.parse((src/'downstream tasks/classification.py').read_text())
    cfg=ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='vae_config' for t in n.targets)))
    cfg.update(sample_size=64,block_out_channels=[32]*6,layers_per_block=1,num_layers=1)
    torch.set_num_threads(4);torch.manual_seed(20261012)
    native=load_author(src/'model')(**cfg);port=CATCodecAdapter(copy.deepcopy(native))
    audit=ActivationAudit('cat_root_api',[('network',native,port.network)],a.output,excluded_modules=('',));rows=[]
    for rates in (None,torch.tensor([1,2,3])):
      for return_dict in (False,True):
       for posterior_sampling in (False,True):
        images=torch.randn(3,3,64,64);native.zero_grad(set_to_none=True);port.zero_grad(set_to_none=True)
        ga=torch.Generator(device='cpu').manual_seed(917);gb=torch.Generator(device='cpu').manual_seed(917)
        audit.begin(len(rows),'root_api');audit.side('native');before=torch.get_rng_state()
        aa=native(images,rates,sample_posterior=posterior_sampling,return_dict=return_dict,generator=ga);after=torch.get_rng_state()
        audit.side('unified');torch.set_rng_state(before)
        bb=port(images,rates,sample_posterior=posterior_sampling,return_dict=return_dict,generator=gb)
        assert type(aa) is type(bb) is tuple and len(aa)==len(bb)==(2 if return_dict else 1)
        assert torch.equal(after,torch.get_rng_state()) and torch.equal(ga.get_state(),gb.get_state())
        for x,y in zip(aa,bb):assert x.shape==y.shape and x.dtype==y.dtype and torch.equal(byte_view(x),byte_view(y))
        la=aa[0].square().mean()+(aa[1] if return_dict else 0);lb=bb[0].square().mean()+(bb[1] if return_dict else 0)
        assert torch.equal(byte_view(la),byte_view(lb))
        audit.side('native','backward_recompute');la.backward();audit.side('unified','backward_recompute');lb.backward()
        audit.parameters(native,port.network);audit.finish_segment()
        rows.append(dict(rates=None if rates is None else rates.tolist(),return_dict=return_dict,sample_posterior=posterior_sampling,generator_state_equal=True,loss_bytes_equal=True))
    traced=audit.close();assert not traced['structural_mismatches'] and all(s['pairs']>0 and s['pairs']==s['byte_exact'] and s['max_abs']==0 for s in traced['totals'].values())
    assert not torch.cuda.is_initialized()
    result=dict(status='passed',method='cat',activation=traced,cases=rows,device='cpu',gpu_used=False,cuda_initialized=False,
        scope='Reduced codec; author default-rate and return_dict API, explicit CPU posterior generator and diagnostic pixel-square+optional-KL adjoint. Not author training loss.',
        official_training_loss_verified=False,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),adapter_sha256=hashlib.sha256((ROOT/'umdiff/compat/cat_codec.py').read_bytes()).hexdigest())
    (a.output/'cat_api.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status='passed',cases=len(rows))))
if __name__=='__main__':main()
