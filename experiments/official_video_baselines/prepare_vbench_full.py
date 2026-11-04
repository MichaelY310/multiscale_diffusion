"""CPU-only installation/weight audit for official VBench's 16 dimensions."""
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import time


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='':
        raise RuntimeError('Prepare scorer assets on CPU only')
    import torch
    from vbench import VBench
    from vbench.utils import init_submodules
    out=Path(os.environ['VBENCH_CACHE_DIR']);out.mkdir(parents=True,exist_ok=True)
    report={'status':'running','dimensions':{},'vbench':importlib.metadata.version('vbench'),
        'torch':torch.__version__,'transformers':importlib.metadata.version('transformers'),
        'compatibility_note':'isolated Python3.12 and Transformers4.44.2; official0.1.5 published4.33.2',
        'quality_result':False,'cuda_context_initialized':False}
    evaluator=VBench(torch.device('cpu'),'unused',str(out/'audit'))
    for dimension in evaluator.build_full_dimension_list():
        row={};begin=time.time()
        try:
            importlib.import_module('vbench.'+dimension)
            row['import']='passed'
        except Exception as exc:row['import']='failed';row['import_error']=repr(exc)
        try:
            modules=init_submodules([dimension],local=True)
            row['submodules']=modules[dimension];row['weights']='prepared'
            if dimension=='aesthetic_quality':
                from vbench.aesthetic_quality import get_aesthetic_model
                get_aesthetic_model(modules[dimension][1])
        except Exception as exc:row['weights']='failed';row['weights_error']=repr(exc)
        row['seconds']=time.time()-begin;report['dimensions'][dimension]=row
        (out/'preparation.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({dimension:row}),flush=True)
    report['files']=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,
        'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in out.rglob('*')
        if p.is_file() and p.suffix in ('.pt','.pth')]
    report['status']='passed' if all(r['import']=='passed' and r['weights']=='prepared'
        for r in report['dimensions'].values()) else 'failed'
    assert not torch.cuda.is_initialized()
    (out/'preparation.json').write_text(json.dumps(report,indent=2))
    if report['status']!='passed':raise RuntimeError('Some official dimensions require repairs; see preparation.json')


if __name__=='__main__':main()
