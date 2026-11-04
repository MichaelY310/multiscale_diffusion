"""CPU synthetic legacy weight-envelope conversion and real sample CLI check."""
import copy, json, os, subprocess, sys, tempfile, hashlib
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
import torch
from umdiff import build
from umdiff.cli import make_method
from umdiff.randomness import RandomStream
from import_legacy_rdd_checkpoint import import_checkpoint


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def eq(a,b):assert a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))

def main():
    out=ROOT/'evidence/rdd_reproduction_20261104/import';out.mkdir(parents=True,exist_ok=True)
    rows=[];rejections=[];torch.set_num_threads(2)
    env={**os.environ,'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
    with tempfile.TemporaryDirectory(prefix='rdd_import_check_') as tmp:
        tmp=Path(tmp)
        for family,name in [('fm','rdd_fm_xl_v2_euler_cfg1p0_noise_filling'),('ddpm','rdd_ddpm_xl_300_300_400_ddpm_cfg1p0_fresh')]:
            c=json.loads((ROOT/'configs/rdd_reproduction'/f'{name}.json').read_text())
            c['model']['args'].update(hidden_size=32,depth=2,num_heads=4)
            if family=='fm':c['process']['steps']=[3,2,4]
            else:c['sampler']['args']['steps']=[3,2,4]
            path=tmp/(family+'.json');path.write_text(json.dumps(c))
            torch.manual_seed(59);net=build('model',c['model'])
            with torch.no_grad():
                for p in net.parameters():p.normal_(0.,.03)
            ema=copy.deepcopy(net)
            with torch.no_grad():
                for p in ema.parameters():p.mul_(.9)
            old=dict(model=net.state_dict(),ema=ema.state_dict(),step=400000,resolutions=[32,16,8])
            if family=='fm':old.update(s_boundaries=[0.,.3,.6,1.],stage_lengths=[300,300,400])
            else:old['stage_lengths']=[300,300,400]
            source=tmp/(family+'_old.pt');torch.save(old,source)
            for weights in ('ema','model'):
                destination=tmp/f'{family}_{weights}.pt'
                p=subprocess.run([sys.executable,str(ROOT/'tools/import_legacy_rdd_checkpoint.py'),'--config',str(path),'--checkpoint',str(source),'--output',str(destination),'--weights',weights],env=env,text=True,capture_output=True,timeout=60)
                assert p.returncode==0,p.stdout+p.stderr
                record=json.loads(p.stdout);converted=torch.load(destination,weights_only=False)
                for k,v in old[weights].items():eq(v,converted['ema'][k]);eq(v,converted['model'][k])
                assert converted['config']['training']['inference_only'] and 'optimizer' not in converted
                sample=tmp/f'{family}_{weights}_sample.pt'
                command=[sys.executable,'-m','umdiff','sample','--checkpoint',str(destination),'--count','1','--labels','17','--seed','85','--device','cpu','--output',str(sample)]
                p=subprocess.run(command,env=env,text=True,capture_output=True,timeout=60);assert p.returncode==0,p.stdout+p.stderr
                got=torch.load(sample,weights_only=True);method=make_method(c);reference=ema if weights=='ema' else net;reference.eval()
                stream=RandomStream(85);initial=stream.normal(torch.empty(1,4,8,8))
                expected=method.sample(reference,initial,stream=stream,model_kwargs={'y':torch.tensor([17])})
                eq(expected,got)
                rows.append(dict(family=family,selected_weights=weights,status='passed',real_import_cli=True,real_sample_cli=True,all_state_tensors_byte_exact=True,sampling_latent_byte_exact=True,source_sha256=sha(source),converted_sha256=sha(destination),sample_sha256=sha(sample)))
            # All invalid inputs must fail without writing a canonical envelope.
            for failure in ('missing_key','extra_key','wrong_shape','wrong_clock','missing_ema'):
                bad=copy.deepcopy(old)
                key=next(iter(bad['ema']))
                if failure=='missing_key':del bad['ema'][key]
                elif failure=='extra_key':bad['ema']['unexpected.weight']=torch.zeros(1)
                elif failure=='wrong_shape':bad['ema'][key]=torch.zeros(1)
                elif failure=='wrong_clock':bad['stage_lengths']=[400,300,300]
                else:del bad['ema']
                candidate=tmp/f'{family}_{failure}_old.pt';torch.save(bad,candidate);destination=tmp/f'{family}_{failure}_converted.pt'
                try:import_checkpoint(path,candidate,destination)
                except (ValueError,RuntimeError) as e:message=str(e)
                else:raise AssertionError('Expected strict rejection: '+failure)
                assert not destination.exists()
                rejections.append(dict(family=family,case=failure,rejected=True,message=message))
    report=dict(status='passed',device='cpu',cases=rows,strict_rejections=rejections,all_temporary_checkpoint_and_sample_payloads_deleted=True,
        scope='Tiny nonzero random DiT and synthetic legacy checkpoint; validates strict EMA/model selection, metadata rejection and actual inference CLI envelope. No real archived checkpoint downloaded or formal quality run.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),ROOT/'tools/import_legacy_rdd_checkpoint.py',ROOT/'umdiff/cli.py']})
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(status='passed',cases=len(rows),rejections=len(rejections))))
if __name__=='__main__':main()
