"""Spatial residual interpolation ablation; sampler noise filling is unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import benchmark_reuse_latency_tiers as bench
from attention_analysis import one_call_cascade_controller as controller_module

OriginalController=controller_module.OneCallCascadeController


class SmoothLiftController(OriginalController):
    def configure(self,*args,**kwargs):
        kwargs['cross_stage_reuse_resize']='trilinear'
        return super().configure(*args,**kwargs)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True)
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--width',type=int,default=1280);p.add_argument('--height',type=int,default=768)
    p.add_argument('--frames',type=int,default=81);p.add_argument('--port',type=int,default=30122)
    p.add_argument('--groups',type=int,default=3);p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--factors',default='spatial221');p.add_argument('--variants',default='fine24,fine28')
    p.add_argument('--vae-precision',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--seed',type=int,default=20660913)
    p.add_argument('--prompt',default='A red fox walking through a snowy forest, cinematic natural light.')
    a=p.parse_args()
    if a.groups<2 or a.warmups<1:p.error('Require paired repeats and warmup')
    controller_module.OneCallCascadeController=SmoothLiftController
    bench.main(a)
    f=Path(a.output)/'metrics.json';m=json.loads(f.read_text())
    m['reuse_resize']='trilinear';m['boundary_resize']='unchanged native noise_filling'
    m['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    f.write_text(json.dumps(m,indent=2))
