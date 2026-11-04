"""Packed backbone parity and cross-example isolation, not a quality claim."""
import argparse
import json
from pathlib import Path
import os
import torch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--gpu', action='store_true')
    p.add_argument('--full-model', type=Path)
    p.add_argument('--precision', choices=['fp32','bf16'], default='bf16')
    args=p.parse_args()
    if args.gpu:
        if not os.environ.get('RDD_JOB_TOKEN'):
            raise RuntimeError('Use the independent guarded GPU launcher')
        device=torch.device('cuda:0'); dtype=torch.bfloat16 if args.precision=='bf16' else torch.float32
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.set_float32_matmul_precision('highest')
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
            raise RuntimeError('CPU test requires CUDA_VISIBLE_DEVICES empty')
        device=torch.device('cpu'); dtype=torch.float32
    from diffusers import WanTransformer3DModel
    from pyramidal_wan_network import PaperWanTransformer
    torch.set_num_threads(4); torch.manual_seed(219)
    if args.full_model:
        dit=WanTransformer3DModel.from_pretrained(args.full_model,subfolder='transformer',
            torch_dtype=torch.float32,local_files_only=True).to(device)
        text_dim=4096; sizes=[(9,16,32),(5,8,16),(3,4,8)]
    else:
        dit=WanTransformer3DModel(num_attention_heads=2,attention_head_dim=24,in_channels=16,
            out_channels=16,text_dim=16,freq_dim=16,ffn_dim=96,num_layers=2).to(device)
        text_dim=16; sizes=[(9,8,16),(5,4,8),(3,2,4)]
    model=PaperWanTransformer(dit,backend='flex' if args.gpu else 'dense').to(device)
    model.gradient_checkpointing=True
    with torch.no_grad():model.first_frame_embedding.normal_(std=.01)
    x=[torch.randn(1,16,*shape,device=device) for shape in sizes]
    z=[torch.tensor([v],device=device) for v in (.25,.75,.97)]
    c=[torch.randn(1,5,text_dim,device=device,dtype=dtype) for _ in sizes]
    ctx=torch.autocast('cuda',dtype=torch.bfloat16,enabled=dtype==torch.bfloat16) if args.gpu else torch.autocast('cpu',enabled=False)
    with ctx:
        packed=model(x,z,c)
        loss=sum(y.float().square().mean() for y in packed)
    loss.backward()
    pg={name:parameter.grad.detach().clone() for name,parameter in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    with ctx:
        reference=model.reference(x,z,c)
        ref_loss=sum(y.float().square().mean() for y in reference)
    ref_loss.backward()
    checks=[]
    limit=.035 if dtype==torch.bfloat16 else 5e-5
    gradient_difference_squared=0.;gradient_reference_squared=0.
    for name,a,b in [(f'output_{i}',a,b) for i,(a,b) in enumerate(zip(packed,reference))]+[
            (f'gradient_{name}',pg[name],parameter.grad) for name,parameter in model.named_parameters()]:
        difference=(a.detach().float()-b.detach().float())
        relative=float(difference.norm()/b.detach().float().norm().clamp_min(1e-7))
        scaled_max=float(difference.abs().max()/b.detach().float().abs().max().clamp_min(1e-7))
        is_gradient=name.startswith('gradient_')
        if is_gradient:
            gradient_difference_squared+=float(difference.double().square().sum())
            gradient_reference_squared+=float(b.detach().double().square().sum())
        # A pure relative test is ill-conditioned for near-zero Q/K bias
        # gradients. FP32 uses torch.testing's default absolute floor (1e-5),
        # plus the original 5e-5 relative bound on the whole gradient vector.
        # Retain every old pure-relative result as an explicit diagnostic.
        absolute_floor=1e-5 if dtype==torch.float32 else 0.
        passes=(relative<=limit and scaled_max<=limit)
        accepted=bool(torch.all(difference.abs() <= absolute_floor+limit*b.detach().float().abs())) if dtype==torch.float32 else passes
        checks.append({'name':name,'relative_l2':relative,'scaled_max_error':scaled_max,
                       'absolute_max_error':float(difference.abs().max()),
                       'reference_abs_mean':float(b.detach().float().abs().mean()),
                       'pure_relative_passed':passes,'passed':accepted})
    with torch.no_grad(),ctx:
        changed=list(c);changed[0]=c[0]+7
        altered=model(x,z,changed)
    isolation=all(torch.equal(a,b) for a,b in zip(packed[1:],altered[1:]))
    gradient_relative=(gradient_difference_squared/max(gradient_reference_squared,1e-30))**.5
    result={'status':'passed' if all(x['passed'] for x in checks) and isolation and gradient_relative<=limit else 'failed',
        'checks':checks,'cross_example_isolation_exact':isolation,'tolerance':limit,
        'all_parameters_checked':len(pg),'first_frame_embedding_gradient_norm':float(pg['first_frame_embedding'].norm()),
        'parameters':sum(p.numel() for p in model.parameters()),'full_pretrained_model':bool(args.full_model),
        'cuda_initialized':torch.cuda.is_initialized(),'bitwise_equal_required':False,
        'precision':str(dtype),
        'absolute_tolerance':1e-5 if dtype==torch.float32 else 0.,
        'whole_gradient_relative_l2':gradient_relative,
        'pure_relative_failures':[x['name'] for x in checks if not x['pure_relative_passed']],
        'scope':'same native Wan modules, packed block-diagonal self/cross attention versus original Wan.forward with first-frame embedding'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='checks'}),flush=True)
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
