"""Layer-placement controls: hold skipped count fixed and retain early blocks.

This wrapper leaves the original tier benchmark and checkpoint behavior intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import benchmark_reuse_latency_tiers as bench

bench.CONFIGS.update({
    'head1_tail5': bench.config(range(1,25)),
    'head2_tail4': bench.config(range(2,26)),
    'head3_tail3': bench.config(range(3,27)),
    'head1_tail5_s075': bench.config(range(1,25),scale=.75),
    'cascade24_s075': dict(masks=bench.OLD_CONFIGS['cascade24'],scale=.75),
    'cascade28_s075': dict(masks=bench.OLD_CONFIGS['cascade28'],scale=.75),
})

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=832);p.add_argument('--height',type=int,default=448)
    p.add_argument('--frames',type=int,default=29);p.add_argument('--port',type=int,default=30093)
    p.add_argument('--groups',type=int,default=4);p.add_argument('--warmups',type=int,default=1)
    p.add_argument('--vae-precision',choices=['fp32','bf16'],default='bf16')
    p.add_argument('--factors',default='temporal221')
    p.add_argument('--variants',default='head1_tail5,head2_tail4,head3_tail3,head1_tail5_s075')
    p.add_argument('--seed',type=int,default=20660913)
    p.add_argument('--prompt',default='A red fox walking through a snowy forest, cinematic natural light.')
    a=p.parse_args()
    if a.groups<2 or a.warmups<1:p.error('Require paired repeats and warmup')
    bench.main(a)
    f=Path(a.output)/'metrics.json';m=json.loads(f.read_text())
    m['wrapper_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    f.write_text(json.dumps(m,indent=2))
