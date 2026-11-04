"""One CPU update at each shipped example's actual shape/model settings."""
import gc
import hashlib
import json
from pathlib import Path
import torch
from umdiff.cli import setup,conditions
from umdiff.registry import build

ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2);directory=ROOT/'configs/cross_modal'
    report={'device':'cpu','formal_training':False,'cases':[]}
    destination=ROOT/'evidence/cross_modal_20261013/config_preflight.json'
    for file in json.loads((directory/'manifest.json').read_text())['configs']:
        path=directory/file;config=json.loads(path.read_text())
        method,model,trainer=setup(config,torch.device('cpu'),19)
        batch={'images':torch.randn(1,*config['image_shape'])}
        if config['model']['name']=='wan_grid':batch['text_embeddings']=torch.randn(1,3,16)
        kwargs,ctx=conditions(config['training'],batch,torch.device('cpu'),1,stream=trainer.stream)
        sampler=build('time_sampler',config['training']['time_sampler'])
        result=trainer.step(batch['images'],None,time_sampler=sampler,model_kwargs=kwargs,context=ctx)
        row={'file':file,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'loss':float(result.loss.detach().mean()),'parameters':sum(p.numel() for p in model.parameters()),
             'nonzero_gradient_parameters':sum(p.grad is not None and bool(p.grad.abs().max()>0) for p in model.parameters()),
             'shape':config['image_shape'],'all_parameters_finite':all(bool(torch.isfinite(p).all()) for p in model.parameters())}
        assert row['all_parameters_finite'] and row['nonzero_gradient_parameters']>0
        report['cases'].append(row);destination.write_text(json.dumps(report,indent=2)+'\n')
        print(file,'PASS',flush=True)
        del result,trainer,model,method;gc.collect()
    report['pass']=True;report['cuda_initialized']=torch.cuda.is_initialized()
    assert not report['cuda_initialized']
    destination.write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
