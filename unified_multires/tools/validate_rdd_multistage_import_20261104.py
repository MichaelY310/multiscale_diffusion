"""Actual CPU CLI import -> umdiff sample with temporary random stage weights."""
import argparse
import ast
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import torch
from umdiff import build
from umdiff.rdd_native import diffusion_backend
from umdiff import rdd_multistage
from import_legacy_rdd_multistage import convert


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(2)
    out=ROOT/'evidence/rdd_reproduction_20261104/ddpm_boundary/multistage_import_v2';out.mkdir(parents=True,exist_ok=True)
    config=ROOT/'configs/rdd_reproduction/rdd_ddpm_multistage_xl_l_b_300_300_400_ddpm_cfg1p0_fresh.json'
    c=json.loads(config.read_text())
    for r,spec in c['model']['args']['models'].items():spec['args'].update(hidden_size=32,depth=2,num_heads=4)
    c['sampler']['args']['steps']=[3,3,3]
    with tempfile.TemporaryDirectory(prefix='rdd_multistage_import_') as folder:
        folder=Path(folder);fixture_config=folder/'config.json';fixture_config.write_text(json.dumps(c))
        torch.manual_seed(824)
        original=build('model',c['model']).eval()
        with torch.no_grad():
            for p in original.parameters():p.normal_(0,.03)
        mapping={}
        for r in (32,16,8):
            i=(32,16,8).index(r);limits=[0,300,600,1000]
            file=folder/f'{r}.pt';torch.save({'ema':original.for_resolution(r).state_dict(),
                'resolution':f'{r}x{r}','stage_lengths':[300,300,400],
                'args':argparse.Namespace(stage_lengths='300,300,400'),
                'T_c':limits[i],'T_c_plus_1':limits[i+1],
                'step':400000 if r!=8 else 600000},file);mapping[str(r)]=file
        envelope=folder/'bundle.pt'
        command=[sys.executable,str(ROOT/'tools/import_legacy_rdd_multistage.py'),'--config',str(fixture_config),
            '--output',str(envelope)]
        for r,path in mapping.items():command+=['--checkpoint',f'{r}={path}']
        imported=subprocess.run(command,text=True,capture_output=True,check=True)
        (out/'import_stdout.log').write_text(imported.stdout);(out/'import_stderr.log').write_text(imported.stderr)
        sample_file=folder/'sample.pt';command=[sys.executable,'-m','umdiff','sample','--checkpoint',str(envelope),
            '--count','1','--labels','17','--seed','4242','--device','cpu','--output',str(sample_file)]
        sampled=subprocess.run(command,text=True,capture_output=True,check=True)
        (out/'sample_stdout.log').write_text(sampled.stdout);(out/'sample_stderr.log').write_text(sampled.stderr)
        actual=torch.load(sample_file,weights_only=True)
        # Independent original stage assembly with fixed conditioning matching
        # the CLI; not the old FID evaluator's automatic random class draw.
        backend=diffusion_backend()
        from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
        source=ROOT/'evidence/rdd_reproduction_20261104/prechange_sources/multiscale_diffusion/2d/sample_fid_unified.py'
        tree=ast.parse(source.read_text());names={'_redraw_upsample','_run_stage'}
        selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
        env={'torch':torch,'_extract_into_tensor':_extract_into_tensor};exec(compile(ast.Module(selected,type_ignores=[]),str(source),'exec'),env)
        limits=[0,300,600,1000];ds=[backend.create_diffusion_with_drift('3',diffusion_steps=1000,
            c=i if i<2 else None,T_c=limits[i],T_c_plus_1=limits[i+1],sample_scale=2) for i in range(3)]
        torch.manual_seed(4242);state=torch.randn(1,4,8,8)
        with torch.no_grad():
            for j,i in enumerate((2,1,0)):
                if j:state,_=env['_redraw_upsample'](result['pred_xstart'].float(),torch.full((1,),2),ds[i],(32,16,8)[i])
                net=original.for_resolution((32,16,8)[i])
                result=env['_run_stage'](ds[i],net.forward,state,{'y':torch.tensor([17])},'cpu','ddpm');state=result['sample']
        assert torch.equal(actual.view(torch.uint8),state.float().view(torch.uint8))
        wrong=folder/'wrong.pt';torch.save({'ema':original.for_resolution(8).state_dict(),'resolution':16},wrong)
        bad=dict(mapping);bad['8']=wrong
        try:convert(fixture_config,bad,folder/'should_not_exist.pt')
        except ValueError as e:rejection=str(e)
        else:raise AssertionError('Wrong resolution metadata was accepted')
        assert not (folder/'should_not_exist.pt').exists()
        metadata_rejections=[]
        for field,value in [('T_c',0),('T_c_plus_1',900),('args',argparse.Namespace(stage_lengths='100,300,600'))]:
            fixture={'ema':original.for_resolution(8).state_dict(),'resolution':'8x8',field:value}
            torch.save(fixture,wrong)
            try:convert(fixture_config,{**mapping,'8':wrong},folder/'should_not_exist.pt')
            except ValueError as error:metadata_rejections.append(str(error))
            else:raise AssertionError('Wrong native metadata accepted')
        cleanup=[{'file':p.name,'bytes':p.stat().st_size,'sha256':sha(p),'removed_after_check':True} for p in folder.glob('*.pt')]
        metadata=json.loads(envelope.with_suffix('.pt.import.json').read_text())
        report={'status':'passed','actual_cli_import':True,'actual_cli_sample':True,'final_latent_byte_exact':True,
            'max_abs':float((actual-state).abs().max()),'latent_shape':list(actual.shape),
            'mixed_stage_steps_preserved':{str(r['resolution']):r['source_step'] for r in metadata['source_stages']},
            'wrong_metadata_rejected':rejection,'inference_only':True,'optimizer_resume_converted':False,
            'interval_and_args_metadata_rejections':metadata_rejections,
            'cleanup':cleanup,'scope':'CPU random native per-grid DiT, complete 3/3/3 DDPM sampling; model import/layout and real CLI, not a quality run.',
            'cuda_initialized':torch.cuda.is_initialized(),'sources':{str(p.relative_to(ROOT.parent)):sha(p) for p in (Path(__file__),ROOT/'tools/import_legacy_rdd_multistage.py',ROOT/'umdiff/rdd_multistage.py',config)}}
    (out/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
