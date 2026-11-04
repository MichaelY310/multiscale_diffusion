#!/usr/bin/env python3
"""Execute SSD's actual bounded TrainLoop, independently match its updates.

Uses real single-rank Gloo, source run_step/backward/AdamW/EMA/LR/save. Does not
claim the private dataset/pretrained model or original CUDA Hydra environment.
"""
import argparse,copy,hashlib,json,os,random,sys,tempfile,time,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'_deps'),str(ROOT/'tools')]
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
import torch
from types import SimpleNamespace
from umdiff import build,native
from umdiff.training import Trainer
from umdiff.engine import TrainingResult
from umdiff.source_objectives import setup
from umdiff.compat.source_fixes import stop_only_loop
from validate_paired_experiments import reduced
from validate_training import nonzero_output
from validate_full_forward_backward import snapshot_rng,restore_rng,fingerprint
from compare_released_profiles import compare_state
from activation_trace import ActivationAudit,byte_view


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2);torch.manual_seed(20261012);np.random.seed(20261012);random.seed(20261012)
    started=time.time();audit=None;report=dict(gpu_used=False,formal_training=False,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        c,_=reduced(json.loads((ROOT/'configs/paired_experiments/ssd_celeba64.json').read_text()))
        model=nonzero_output(build('model',c['model']));port_model=copy.deepcopy(model)
        fixture=torch.randn(2,*c['image_shape']);_,_,_,_,_=setup('ssd',c,fixture)
        from guided_diffusion import scale_space_diffusion as sd,train_util,dist_util,logger
        from guided_diffusion.resample import UniformSampler
        from guided_diffusion.nn import update_ema
        mapping={int(k):v for k,v in c['batch']['args']['unet_res_layer_map'].items()}
        settings=dict(resolution_schedule='ConvexDecay_0.5',t_res_sampling_mode='one_res_many_t_except_one_t_for_res_change',
            multires_training_mode='flexi_unet',unet_res_layer_map=mapping,scale_min=8)
        diffusion=sd.ScaleSpaceDiffusion(betas=np.array(c['batch']['args']['betas']),model_mean_type=sd.ModelMeanType.START_X_RS,
            model_var_type=sd.ModelVarType.FIXED_SMALL,loss_type=sd.LossType.MSE,ssd_config=settings,
            resolutions_array=[8,16,32,64],mse_loss_weight_type='min_snr_5',rescale_timesteps=False)
        for op in diffusion.ops.values():op.H_out,op.W_out=op.size_out
        with tempfile.TemporaryDirectory(prefix='umdiff_native_ssd_') as directory:
            directory=Path(directory);torch.distributed.init_process_group('gloo',init_method='file://'+str(directory/'rendezvous'),rank=0,world_size=1)
            sd.dist=torch.distributed;dist_util.dev=lambda:torch.device('cpu')
            logger.configure(dir=str(directory/'author'),format_strs=['log'])
            def data():
                while True:yield fixture,{}
            loop=stop_only_loop(train_util.TrainLoop,4)(model=model,diffusion=diffusion,data=data(),
                batch_size=2,microbatch=1,lr=1e-4,ema_rate=.9999,log_interval=999,save_interval=999,
                resume_checkpoint='',use_fp16=False,weight_decay=0.,lr_anneal_steps=10,models_dir=str(directory/'author'),ssd_config=settings)
            class Port:
                config=c
                def training(self,net,x,t,stream,noise=None,model_kwargs=None,active=None,context=None):
                    _,_,fn,_,_=setup('ssd',c,x)
                    return TrainingResult(fn(net,context['step']),None,None,None,{})
            trainer=Trainer(Port(),port_model,lr=1e-4,ema_decay=.9999,randomness='native_global',
                accumulation_steps=2,accumulation_normalization='sum',optimizer_parameters='all',
                lr_schedule={'name':'linear','total_steps':10,'update_timing':'after_step'})
            audit=ActivationAudit('ssd_native_loop',[('network',model,port_model)],a.output/'layers');rows=[]
            raw=loop.run_step
            def paired(batch,cond):
                step=loop.step;audit.begin(step);audit.side('native');before=snapshot_rng()
                mean_loss=raw(batch,cond);after=snapshot_rng()
                audit.side('unified');restore_rng(before);result=trainer.step(batch,None)
                audit.parameters(model,port_model);audit.finish_segment()
                native_ema=loop.mp_trainer.master_params_to_state_dict(loop.ema_params[0])
                checks={k:compare_state(x,y) for k,x,y in [('model',model.state_dict(),port_model.state_dict()),
                    ('optimizer',loop.opt.state_dict(),trainer.optimizer.state_dict()),('ema',native_ema,trainer.ema.state_dict())]}
                if any(v['exact']!=v['tensors'] for v in checks.values()):raise AssertionError('Native lifecycle state differs')
                if fingerprint(after)!=fingerprint(snapshot_rng()):raise AssertionError('Native lifecycle RNG differs')
                rows.append(dict(step=step,state=checks,lr=loop.opt.param_groups[0]['lr'],rng_byte_exact=True,
                    native_logged_loss=mean_loss.item(),unified_logged_loss=result.loss.mean().item()))
                restore_rng(after);return mean_loss
            loop.run_step=paired;loop.run_loop()
            if loop.step!=4 or loop.lr_anneal_steps!=10:raise AssertionError('Stop cap changed schedule or update count')
            saved=directory/'author/model_000004.pt'
            if not saved.exists():raise AssertionError('Source final checkpoint absent')
            checkpoint=torch.load(saved,weights_only=False)
            if compare_state(checkpoint['model'],model.state_dict())['exact']!=len(model.state_dict()):raise AssertionError('Source saved model differs')
            # Explicitly inspect author's limitations instead of certifying a
            # model-only final file as exact resumption of the training process.
            report['native_checkpoint_contract']=dict(keys=list(checkpoint),step=checkpoint['step'],
                final_optimizer_saved=(directory/'author/opt_000004.pt').exists(),rng_saved='rng' in checkpoint,
                reader_saved='reader' in checkpoint,periodic_save_occurs_before_step_increment=True)
            activation=audit.close();audit=None
            if activation['structural_mismatches'] or any(v['pairs']!=v['byte_exact'] for v in activation['totals'].values()):raise AssertionError('Native loop raw layer/gradient mismatch')
            report.update(status='passed',steps=rows,activation=activation,stop_adapter=loop.compatibility_evidence,
                source_loop_executed=True,source_hydra_main_executed=False,
                scope='Actual SSD TrainLoop4 updates with original ten-step LR;reduced random author backbone/64grid,synthetic data,realGloo rank0,CPU FP32;strict raw layer/gradient/state comparisons. Native final checkpoint omissions explicitly detected.')
            audit=None;torch.distributed.destroy_process_group()
    except Exception:
        report.update(status='failed',failure=traceback.format_exc());print(report['failure'],flush=True)
        if torch.distributed.is_initialized():torch.distributed.destroy_process_group()
    if audit is not None:report['activation']=audit.close()
    report.update(seconds=time.time()-started,source_executions=native.AUDIT,cuda_initialized=torch.cuda.is_initialized())
    (a.output/'ssd_native_loop.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:report[k] for k in ('status','seconds')}),flush=True)
    return report['status']!='passed'
if __name__=='__main__':raise SystemExit(main())
