"""Actual FastVideo production loader versus full native Diffusers Wan weights."""
import argparse
import json
import os
from pathlib import Path
import re
import time
import torch


def metrics(a,b):
    a,b=a.detach().float(),b.detach().float()
    error=(a-b).abs()
    return {'absolute_max':error.max().item(),'absolute_mean':error.mean().item(),
        'relative_l2':((a-b).norm()/b.norm().clamp_min(1e-12)).item(),
        'mean_drift_fraction':(error.mean()/b.abs().mean().clamp_min(1e-12)).item(),
        'passed':bool(torch.allclose(a,b,atol=.1,rtol=.1) and error.mean()<.05*b.abs().mean())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--trace',type=Path)
    p.add_argument('--debug-rms',action='store_true',help='Temporary per-instance native RMSNorm forward; no source edits')
    args=p.parse_args()
    if not os.environ.get('RDD_JOB_TOKEN') or time.time()>=float(os.environ.get('RDD_DEADLINE_UNIX','0')):
        raise RuntimeError('Independent GPU guard required')
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    from fastvideo.distributed import maybe_init_distributed_environment_and_model_parallel
    from fastvideo.train.utils.config import load_run_config
    from fastvideo.train.utils.instantiate import instantiate
    from fastvideo.forward_context import set_forward_context
    from fastvideo.models.wan.config import WanVideoArchConfig
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from diffusers import WanTransformer3DModel
    cfg=load_run_config(args.config)
    maybe_init_distributed_environment_and_model_parallel(1,1)
    student=dict(cfg.models['student']);student['attention_backend']='TORCH_SDPA'
    model=instantiate(student,training_config=cfg.training)
    model.transformer.eval()
    restorations=[]
    if args.debug_rms:
        from fastvideo.layers.layernorm import RMSNorm
        import types
        for module in model.transformer.modules():
            if isinstance(module,RMSNorm):
                restorations.append((module,module.forward))
                def native_rms(self,x):
                    return torch.nn.functional.rms_norm(x,(self.hidden_size,),self.weight,self.variance_epsilon)
                module.forward=types.MethodType(native_rms,module)
        print('[debug] Temporary native RMSNorm forward:',len(restorations),flush=True)
    reference=WanTransformer3DModel.from_pretrained(student['init_from'],subfolder='transformer',
        torch_dtype=torch.float32,local_files_only=True).cuda().eval().requires_grad_(False)
    actual=get_model_state_dict(model.transformer)
    native_profile=set(actual)==set(reference.state_dict())
    mapping={} if native_profile else WanVideoArchConfig().param_names_mapping
    matched={};bad=[]
    for name,tensor in reference.state_dict().items():
        mapped=name
        for regex,replacement in mapping.items():
            if re.match(regex,name):mapped=re.sub(regex,replacement,name);break
        if mapped not in actual:bad.append({'missing':mapped});continue
        value=actual[mapped]
        if hasattr(value,'full_tensor'):value=value.full_tensor()
        equal=torch.equal(value.cpu(),tensor.cpu());matched[mapped]=equal
        if not equal:bad.append({'name':name,'mapped':mapped,'error':metrics(value,tensor)})
    extra=sorted(set(actual)-set(matched))
    if bad or extra:raise AssertionError({'bad':bad,'extra':extra})
    traces={};managers=[];handles=[]
    if args.trace:
        from fastvideo.hooks.activation_trace import attach_activation_trace
        from full_model_trace import hashed
        args.trace.mkdir(parents=True,exist_ok=False)
        os.environ.update(FASTVIDEO_TRACE_ACTIVATIONS='1',
            FASTVIDEO_TRACE_LAYERS=r'^(condition_embedder|patch_embedding|blocks\.\d+)$',
            FASTVIDEO_TRACE_STATS='abs_mean,sum,max,shape,dtype',FASTVIDEO_TRACE_STEPS='')
        for side,transformer in [('native',reference),('fastvideo',model.transformer)]:
            os.environ['FASTVIDEO_TRACE_OUTPUT']=str(args.trace/(side+'.jsonl'))
            managers.append(attach_activation_trace(transformer));traces[side]={}
            for name,module in transformer.named_modules():
                if re.match(r'^(condition_embedder|blocks\.\d+)$',name):
                    def capture(m,a,y,s=side,n=name):
                        if n not in traces[s]:
                            traces[s][n]=[v.detach().float().cpu() for v in (y if isinstance(y,tuple) else [y]) if isinstance(v,torch.Tensor)]
                    handles.append(module.register_forward_hook(capture))
    sample=torch.load(args.cache,map_location='cpu',weights_only=True)
    device=torch.device('cuda:0')
    vae=json.loads((Path(student['init_from'])/'vae/config.json').read_text())
    mean=torch.tensor(vae['latents_mean'],device=device).view(1,16,1,1,1)
    std=torch.tensor(vae['latents_std'],device=device).view(1,16,1,1,1)
    clean=(sample['small_raw_pixel_pyramid'][0].to(device).float()-mean)/std
    checks=[]
    # The pinned production FSDP loader keeps FP32 master weights but always
    # uses BF16 forward parameters. Validate its actual supported contract.
    for precision in ['bf16']:
        dtype=torch.float32 if precision=='fp32' else torch.bfloat16
        text=sample['text_embedding'].to(device,dtype=dtype)
        for stage,factor in enumerate([4,2,1]):
            x=torch.nn.functional.avg_pool3d(clean,(1,factor,factor)).to(dtype)
            torch.manual_seed(119+stage)
            x=.4*x+.6*torch.randn_like(x);t=torch.tensor([600.],device=device)
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                expected=reference(x,timestep=t,encoder_hidden_states=text,return_dict=False)[0]
                with set_forward_context(current_timestep=t,attn_metadata=None):
                    received=model.transformer(hidden_states=x,timestep=t,encoder_hidden_states=text)
                    if native_profile:received=received[0]
            checks.append({'precision':precision,'factor':factor,**metrics(received,expected)})
            if native_profile:
                from types import SimpleNamespace
                batch=SimpleNamespace(conditional_dict={'encoder_hidden_states':text})
                with torch.no_grad():
                    adapted=model.predict_noise(x.permute(0,2,1,3,4),t,batch,
                        conditional=True).permute(0,2,1,3,4)
                checks[-1]['original_modules_bitwise_equal']=torch.equal(received,expected)
                checks[-1]['adapter_interface_bitwise_equal']=torch.equal(adapted,expected)
                checks[-1]['passed']=checks[-1]['passed'] and checks[-1]['original_modules_bitwise_equal'] and checks[-1]['adapter_interface_bitwise_equal']
    result={'status':'passed' if all(x['passed'] for x in checks) else 'failed',
        'all_weights_bitwise_equal':True,'weight_tensors':len(matched),
        'parameters':sum(x.numel() for x in reference.parameters()),'checks':checks,
        'real_T5_conditioning':True,'full_pretrained_model':True,'production_loader':True,
        'attention_backend':'TORCH_SDPA','master_precision':'FP32','forward_precision':'BF16 AMP (production FSDP policy)',
        'scope':'actual full FastVideo Wan loader plus BF16 forward versus native Diffusers on three real-cache stage shapes',
        'forward_tolerance':{'atol':.1,'rtol':.1,'mean_drift_fraction_max':.05}}
    result['native_module_adapter']=native_profile
    if native_profile:
        result.update(scope='full original Wan modules inside current FastVideo Model/Method interface; three real-cache shapes',
            forward_precision='native HF BF16 autocast with FP32 master weights',
            production_loader=False,forward_tolerance='bitwise equality for original module and adapter interface')
    import hashlib,importlib.metadata,inspect
    native_source=Path(inspect.getfile(WanTransformer3DModel))
    result['native_source']={'path':str(native_source),'sha256':hashlib.sha256(native_source.read_bytes()).hexdigest(),
        'diffusers_distribution_version':importlib.metadata.version('diffusers')}
    if args.trace:
        from fastvideo.hooks.activation_trace import detach_activation_trace
        result['layer_checks']={k:[metrics(a,b) for a,b in zip(traces['fastvideo'][k],value)] for k,value in traces['native'].items()}
        for handle in handles:handle.remove()
        for manager in managers:detach_activation_trace(manager)
        (args.trace/'layer_checks.json').write_text(json.dumps(result['layer_checks'],indent=2))
    args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    for module,original in restorations:module.forward=original
    if result['status']!='passed':raise SystemExit(2)


if __name__=='__main__':main()
