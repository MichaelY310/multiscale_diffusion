"""CPU independent-model FM: original training order, full sampling and import."""
from copy import deepcopy
import argparse, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
import torch
from umdiff import build
from umdiff.cli import make_method
from umdiff.training import Trainer
from umdiff.randomness import RandomStream
from activation_trace import ActivationAudit
from validate_rdd_fm_native_20261104 import source,matchers,exact,verify_audit,SamplingHashes,digest
from import_legacy_rdd_multistage import convert

OUT=ROOT/'evidence/rdd_reproduction_20261104/fm_multistage'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2)
    c=json.loads((ROOT/'configs/rdd_reproduction/rdd_fm_multistage_xl_l_b_euler_cfg1p0_fresh.json').read_text())
    c['training'].pop('input_transform',None)
    for spec in c['model']['args']['models'].values():spec['args'].update(hidden_size=32,depth=2,num_heads=4)
    torch.manual_seed(622);native=build('model',c['model']).train()
    with torch.no_grad():
        for p in native.parameters():p.normal_(0,.03)
    port=deepcopy(native);ema=deepcopy(native).eval().requires_grad_(False)
    trainer=Trainer(make_method(c),port,**c['training']['optimizer'])
    opts={r:torch.optim.AdamW(native.for_resolution(r).parameters(),lr=1e-4,weight_decay=0.) for r in (32,16,8)}
    old,helpers=source();fm=matchers(old,c['process']['bounds'])
    audit=ActivationAudit('fm_three_independent_models',[(str(r),native.for_resolution(r),port.for_resolution(r)) for r in (32,16,8)],OUT)
    rows=[]
    for step in range(4):
        images=torch.randn(2,4,32,32);labels=torch.tensor([17,23]);rng=torch.get_rng_state().clone()
        audit.begin(step);audit.side('native');losses=[]
        # Match old trainer: each independent stage backward/step/EMA immediately.
        for i,r in enumerate((32,16,8)):
            x=images if i==0 else helpers['downsample_latent'](images.detach(),r)
            net=native.for_resolution(r);loss=fm[i].training_losses(net,x,model_kwargs={'y':labels})['loss'].mean()
            opts[r].zero_grad();loss.backward();opts[r].step();helpers['update_ema'](ema.for_resolution(r),net)
            losses.append(loss.detach())
        expected_rng=torch.get_rng_state().clone();torch.set_rng_state(rng);audit.side('unified')
        got=trainer.step(images,None,model_kwargs={'y':labels})
        for r in (32,16,8):audit.parameters(native.for_resolution(r),port.for_resolution(r),role=str(r))
        audit.finish_segment();exact(sum(losses),got.loss,'loss');exact(expected_rng,torch.get_rng_state(),'rng')
        exact(native.state_dict(),port.state_dict(),'weights');exact(ema.state_dict(),trainer.ema.state_dict(),'ema')
        for r in (32,16,8):
            for a,b in zip(native.for_resolution(r).parameters(),port.for_resolution(r).parameters()):exact(opts[r].state.get(a,{}),trainer.optimizer.state.get(b,{}),'optimizer')
        rows.append({'step':step,'all_byte_exact':True})
    layers=verify_audit(audit);(OUT/'LAYERS.json').write_text(json.dumps(layers,indent=2)+'\n')
    net=deepcopy(ema).eval();samples=[]
    # Explicit old stage loop uses each original child directly; new loop routes
    # through the unified adapter. Compare all executed child module boundaries.
    for boundary in ('fresh','noise_filling'):
      for solver in ('euler','midpoint'):
       for cfg in (1.,1.5):
        cc=deepcopy(c);cc['process'].update(boundary=boundary,solver=solver,cfg_scale=cfg,steps=[80,80,90] if solver=='euler' else [40,40,45])
        method=make_method(cc);kwargs={'y':torch.tensor([17])};seed=912
        recorders=[SamplingHashes(net.for_resolution(r)) for r in (32,16,8)]
        torch.manual_seed(seed);state=torch.randn(1,4,8,8)
        for j,i in enumerate((2,1,0)):
            r=(32,16,8)[i];f=fm[i]
            if j:
                state=old.noise_filling_transition_fm(state,clean,f.s_k,r) if boundary=='noise_filling' else old.upsample_with_noise_fm(clean,f.s_k,r)
            result=f.sample_ode(net.for_resolution(r),state,cc['process']['steps'][j],model_kwargs=kwargs,cfg_scale=cfg,solver=solver)
            state,clean=result['sample'],result['pred_x1']
        expected_rng=torch.get_rng_state().clone();records_a=[o.close() for o in recorders]
        recorders=[SamplingHashes(net.for_resolution(r)) for r in (32,16,8)]
        stream=RandomStream(seed);actual=method.sample(net,stream.normal(torch.empty(1,4,8,8)),stream=stream,model_kwargs=kwargs)
        records_b=[o.close() for o in recorders]
        exact(clean,actual,'sample');exact(expected_rng,stream.generator.get_state(),'sample_rng');assert records_a==records_b
        samples.append({'boundary':boundary,'solver':solver,'cfg':cfg,'steps_coarse_to_fine':cc['process']['steps'],'nfe':method.last_sample_metrics['nfe'],'layer_pairs':sum(map(len,records_a)),'all_byte_exact':True,'latent_sha256':digest(actual),'trace_sha256':hashlib.sha256(json.dumps(records_a,sort_keys=True).encode()).hexdigest()})
    cleanups=[];rejections=[]
    with tempfile.TemporaryDirectory(prefix='rdd_fm_multistage_import_') as temp:
        temp=Path(temp);cc=deepcopy(c);cc['process']['steps']=[3,3,3];p=temp/'config.json';p.write_text(json.dumps(cc));mapping={}
        for r in (32,16,8):
            file=temp/f'{r}.pt';torch.save({'ema':net.for_resolution(r).state_dict(),'resolution':f'{r}x{r}','args':argparse.Namespace(stage_lengths='300,300,400')},file);mapping[str(r)]=file
        envelope=temp/'bundle.pt';convert(p,mapping,envelope)
        output=temp/'sample.pt';subprocess.run([sys.executable,'-m','umdiff','sample','--checkpoint',str(envelope),'--count','1','--labels','17','--seed','912','--device','cpu','--output',str(output)],capture_output=True,text=True,check=True)
        stream=RandomStream(912);expected=make_method(cc).sample(net,stream.normal(torch.empty(1,4,8,8)),stream=stream,model_kwargs=kwargs)
        exact(expected,torch.load(output,weights_only=True),'real_cli_sample')
        for value in ('16x8','16x16'):
            bad=temp/'bad.pt';torch.save({'ema':net.for_resolution(8).state_dict(),'resolution':value},bad)
            try:convert(p,{**mapping,'8':bad},temp/'rejected.pt')
            except ValueError as error:rejections.append(str(error))
            else:raise AssertionError('Invalid resolution accepted')
        bad=temp/'bad.pt';torch.save({'ema':net.for_resolution(8).state_dict(),'resolution':'8x8','args':argparse.Namespace(stage_lengths='400,300,300')},bad)
        try:convert(p,{**mapping,'8':bad},temp/'rejected.pt')
        except ValueError as error:rejections.append(str(error))
        else:raise AssertionError('Wrong FM clock accepted')
        for file in temp.glob('*.pt'):cleanups.append({'file':file.name,'bytes':file.stat().st_size,'sha256':sha(file),'removed':True})
    files=[Path(__file__),ROOT/'umdiff/rdd_fm_native.py',ROOT/'umdiff/rdd_multistage.py',ROOT/'tools/import_legacy_rdd_multistage.py',ROOT.parent/'2d/train_RDD_FM_imagenet256.py']
    report={'status':'passed','training':{'updates':rows,'totals':layers['totals'],'three_sequential_optimizers_vs_joint_optimizer_byte_exact':True},'sampling':samples,'import':{'square_string_resolution':True,'argparse_clock_metadata':True,'actual_sample_cli_byte_exact':True,'rejections':rejections,'cleanup':cleanups},'scope':'CPU random width32 depth2 per-stage DiT, full32/16/8 grids. Original sequential independent-model training versus unified summed loss, original stage sampling backend and independent stage loops. No formal historical FM-multistage FID was found.','sources':{str(p.relative_to(ROOT.parent)):sha(p) for p in files},'cuda_initialized':torch.cuda.is_initialized()}
    (OUT/'SUMMARY.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))

if __name__=='__main__':main()
