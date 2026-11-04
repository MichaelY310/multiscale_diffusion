"""Cost-only paired RDD+VSA baseline: real denoise+VAE, NO media export.

    The image and video configurations are generated from the same fractions.
    This does not train or make claims about video generation quality.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from diffusion.cross_scale_reuse_config import layer_masks, description
import benchmark_reuse_latency_tiers as benchmark


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='/scr/yuheng/DevOpsBench/multiscale_diffusion')
    p.add_argument('--output', required=True)
    p.add_argument('--variants', default='high_heavy,balanced')
    p.add_argument('--groups', type=int, default=4)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--port', type=int, default=30261)
    p.add_argument('--width', type=int, default=1280)
    p.add_argument('--height', type=int, default=768)
    p.add_argument('--frames', type=int, default=61)
    p.add_argument('--factors', default='temporal421')
    p.add_argument('--seed', type=int, default=20660913)
    args = p.parse_args()
    args.vae_precision = 'bf16'
    args.no_export_media = True
    args.allowed_gpus = tuple(str(i) for i in range(8))  # User authorized both 4-GPU groups for this task.
    args.prompt = 'A red fox walking through a snowy forest, cinematic natural light.'
    for family in args.variants.split(','):
        benchmark.CONFIGS[family] = dict(masks={k: list(v) for k,v in layer_masks(family,30).items() if v}, scale=1.)
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    (out/'image_video_mapping.json').write_text(json.dumps({f:{'image':description(f,24),'video':description(f,30)} for f in args.variants.split(',')}, indent=2))
    benchmark.main(args)
    metrics=json.loads((out/'metrics.json').read_text())
    gate={'complete':True,'baseline':'same shape schedule RDD + native VSA20; NO reuse',
          'criterion':'mean and bootstrap lower bound of paired sample+VAE latency reduction >=20%',
          'shared_gpu':True,'exported_media':False,'candidates':{}}
    for entry in metrics['entries']:
        r=entry['reductions']['total']
        gate['candidates'][entry['variant']]={'passed':r['mean']>=20 and r['lower']>=20,'reduction':r}
    (out/'speed_gate.json').write_text(json.dumps(gate,indent=2))
    print(json.dumps(gate),flush=True)
