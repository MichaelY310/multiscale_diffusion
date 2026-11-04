#!/usr/bin/env python3
"""Full CelebA64 SSD backbone, every constant/transition resolution path on CPU."""
import argparse,copy,hashlib,json,os,sys,traceback,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tools'),str(ROOT/'_deps')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
import torch
from umdiff import build,native
from umdiff.source_objectives import setup
from umdiff.time_sampling import SSDTimes
from activation_trace import ActivationAudit,byte_view
from validate_training import nonzero_output
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint,author_backward
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=4);args=p.parse_args()
    torch.set_num_threads(args.threads);torch.manual_seed(20261012);torch.use_deterministic_algorithms(True)
    path=ROOT/'configs/paired_experiments/ssd_celeba64.json';cfg=json.loads(path.read_text())
    a=nonzero_output(build('model',cfg['model']));b=copy.deepcopy(a);images=torch.randn(2,*cfg['image_shape'])
    _,original,port,_,notes=setup('ssd',cfg,images)
    from guided_diffusion.resample import UniformSampler
    resolutions=cfg['batch']['args']['resolutions'];selected={}
    for t,r in enumerate(resolutions):selected.setdefault((resolutions[max(t-1,0)],r),t)
    audit=ActivationAudit('ssd_all_resolution_edges',[('network',a,b)],args.output.parent/'layers');rows=[];failure=None;started=time.time()
    try:
        for step,((rs,rt),index) in enumerate(selected.items()):
            t=torch.full((len(images),),index,dtype=torch.long);weight=torch.ones(len(images))
            def source_times(*a,**kw):return t,weight,None
            def port_times(*a,**kw):return t,{'importance_weights':weight}
            a.train();b.train();a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True);before=snapshot_rng()
            audit.begin(step);audit.side('native')
            with patch.object(UniformSampler,'sample',source_times):la=original(a,step).mean()
            audit.side('native','backward_recompute');author_backward('ssd',la,images,a);after=snapshot_rng()
            restore_rng(before);audit.side('unified')
            with patch.object(SSDTimes,'__call__',port_times):lb=port(b,step).mean()
            audit.side('unified','backward_recompute');lb.backward();audit.parameters(a,b);audit.finish_segment()
            assert torch.equal(byte_view(la),byte_view(lb)) and fingerprint(after)==fingerprint(snapshot_rng())
            assert all(v['pairs']==v['byte_exact'] for v in audit.totals.values()) and not audit.structural_mismatches
            rows.append(dict(rs=rs,rt=rt,timestep=index,loss_byte_exact=True,post_backward_rng_exact=True))
            print(json.dumps(rows[-1]),flush=True)
    except Exception:failure=traceback.format_exc()
    result=dict(status='failed' if failure else 'passed',device='cpu',gpu_used=False,full_released_architecture=True,config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),steps=rows,activation=audit.close(),failure=failure,seconds=time.time()-started,source_executions=native.AUDIT,scope='Fixed timestep fixtures select every resolution edge; native time draw alone bypassed explicitly on BOTH sides. Source loss and backward remain unchanged; full64 backbone, random weights, batch2 synthetic input. No native CLI or pretrained-quality claim.')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],seconds=result['seconds'],failure=failure)),flush=True)
    return bool(failure)
if __name__=='__main__':raise SystemExit(main())
