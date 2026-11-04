#!/usr/bin/env python3
"""New offline routes: local weight reload plus real fresh-process infer CLI."""
import argparse,hashlib,json,os,subprocess,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import torch
from umdiff.workflows import run,local_module
from validate_fresh_process_lifecycle import manifest


def main():
    p=argparse.ArgumentParser();p.add_argument('route',choices=['mrflow_qwen','speed_pixelgen','speed_wan']);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    started=time.time();torch.set_num_threads(2);report=dict(gpu_used=False,formal_training=False,route=a.route)
    try:
        c=json.loads((ROOT/f'configs/inference_only/{a.route}.json').read_text());shape=c['synthetic_embedding_shape'];g=torch.Generator().manual_seed(711)
        pos=torch.randn(shape,generator=g);neg=torch.randn(shape,generator=g)
        if a.route=='mrflow_qwen':embeddings=dict(prompt_embeds=pos,prompt_embeds_mask=torch.ones(shape[:2],dtype=torch.long),negative_prompt_embeds=neg,negative_prompt_embeds_mask=torch.ones(shape[:2],dtype=torch.long))
        elif a.route=='speed_pixelgen':embeddings=dict(cond=pos,uncond=neg)
        else:embeddings=dict(ctx_cond=pos,ctx_uncond=neg)
        expected=run(c,embeddings,seed=42)
        with tempfile.TemporaryDirectory(prefix='umdiff_new_infer_cli_') as tmp:
            tmp=Path(tmp);torch.manual_seed(42);weights={};dtype=getattr(torch,c.get('dtype','float32'))
            for key in ('model','vae','sr_model'):
                if key not in c:continue
                module=local_module(c[key],None,'cpu',torch.float32 if key=='sr_model' or (key in ('model','vae') and a.route=='speed_wan') else dtype,True)
                path=tmp/(key+'.pt');torch.save(module.state_dict(),path);weights[key]=str(path)
            c.update(synthetic_fixture=False,checkpoints=weights);actual=run(c,embeddings,seed=42)
            if manifest(expected)!=manifest(actual):raise AssertionError('Local state-dict reload differs')
            config=tmp/'config.json';config.write_text(json.dumps(c));torch.save(embeddings,tmp/'embeddings.pt')
            cmd=[sys.executable,'-m','umdiff.cli','infer','--config',str(config),'--embeddings',str(tmp/'embeddings.pt'),'--output',str(tmp/'output.pt'),'--seed','42','--device','cpu']
            with (a.output/(a.route+'.log')).open('w') as log:proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=180)
            if proc.returncode:raise RuntimeError('Infer CLI failed')
            emitted=torch.load(tmp/'output.pt',weights_only=True)
            if manifest(emitted)!=manifest(expected):raise AssertionError('Fresh CLI output differs')
            files=sorted(tmp.glob('output*.png'))
            if not files:raise AssertionError('Infer CLI did not export PNG/frame output')
            report.update(status='passed',local_weights_reload_byte_exact=True,fresh_cli_byte_exact=True,
                output_manifest=manifest(emitted),png_exports=[dict(name=f.name,bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],
                config=c,command=cmd,temporary_payloads_deleted=True,
                scope='Configuration/registry/local state-dict ownership and actual infer CLI,CPU random fixture modules/synthetic supplied embeddings. Wan CPU AMP/SDPA substitutions and Diffusers Wan decoder are explicit,not pretrained native Wan/MP4 certification.')
    except Exception:report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
    report.update(seconds=time.time()-started,harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cuda_initialized=torch.cuda.is_initialized())
    (a.output/(a.route+'.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('route','status','seconds')}),flush=True);return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
