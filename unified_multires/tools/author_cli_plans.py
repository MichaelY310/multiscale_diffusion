"""Real author entrypoints and explicit gaps relative to a matched recipe.

These command arguments are references, never certified as reproducing a best
row when the release or lifecycle cannot implement the requested contract.
"""
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]

def commands(entry,contract):
    paper=entry['paper'];c=contract['controls'];config=contract.get('resolved_config_path',str(ROOT/'configs/paired_experiments'/entry['config']))
    data=contract['data'] or '@data@';assets=contract['assets'];source=ROOT/'sources'/paper
    batch=c.get('global_batch');micro=c.get('micro_batch');updates=c.get('budget')
    native=[sys.executable];cwd=source;missing=[]
    if paper=='rdd':
        cwd=ROOT.parent/'2d';model='DiT-L/2' if '_l_' in entry['id'] else 'DiT-XL/2'
        native+=['-m','torch.distributed.run','--standalone','--nproc_per_node=1',str(cwd/'train_RDD_imagenet256_unified.py'),
          '--model',model,'--data-source','imagenet','--data-path',data,'--results-dir','@original_output@','--global-batch-size',str(batch),
          '--micro-batch-size',str(micro),'--global-seed',str(c['seed']),'--resolutions','32,16,8',
          '--stage-lengths','300,300,400','--diffusion-steps','1000','--vae-path',assets.get('vae','@vae@')]
        if isinstance(updates,int):native+=['--max-train-steps',str(updates)]
        else:missing.append('Historical epochs do not identify an exact continuation/update budget.')
    elif paper=='ppflow':
        levels=3 if entry['id'].endswith('_3') else 2
        native+=['-m','torch.distributed.run','--standalone','--nproc_per_node=1','train.py',
          '--model','PPFlow_XL_'+str(levels),'--patch-level',str(levels),'--data-path',data,
          '--global-batch-size',str(batch),'--global-seed',str(c['seed']),
          '--initialize_from',assets.get('sit','@sit@'),'--results-dir','@original_output@']
        missing+=['Author CLI has no accumulation/max-update/local-VAE-path flags matching this contract.']
    elif paper in ('dimr','dctdiff'):
        if paper=='dimr':name=entry['id'].removeprefix('dimr_')
        else:name=entry['id'].removeprefix('dctdiff_')
        native+=['train.py','--config',str(source/'configs'/(name+'.py')),'--workdir','@original_output@',
          '--config.train.batch_size='+str(batch),'--config.optimizer.lr='+str(c['optimizer']['lr']),
          '--config.optimizer.weight_decay='+str(c['optimizer'].get('weight_decay',0.)),
          '--config.dataset.path='+data,'--config.seed='+str(c['seed'])]
        if isinstance(updates,int):native+=['--config.train.n_steps='+str(updates)]
        missing+=['Source Accelerator/DataLoader/seed/worker/accumulation and asset/evaluation lifecycle are not matched to the unified CLI.']
    elif paper=='dfm':
        native+=['train.py','--exp-name',entry['id'],'--model','DFM-SiT-B/2' if entry['id']=='dfm_sit_b_released_yaml' else 'DFM-SiT-XL/2','--data-dir',data,'--batch-size',str(micro),
          '--gradient-accumulation-steps',str(c.get('accumulation')),'--learning-rate',str(c['optimizer']['lr']),
          '--seed',str(c['seed']),'--output-dir','@original_output@']
        if isinstance(updates,int):native+=['--max-train-steps',str(updates)]
        missing+=['Select the exact SiT config, VAE/teacher/latent data and source Accelerate lifecycle; original paper FLUX/DiT branch differs.']
    elif paper=='pixelflow':
        native+=['-m','torch.distributed.run','--standalone','--nproc_per_node=1','train.py',
          str(source/'configs/pixelflow_xl_c2i.yaml'),'--output-dir','@original_output@']
        missing+=['Requires a source YAML overlay; epoch-only source has no matching accumulation/max-update controls; best training budget is unknown.']
    elif paper=='lapflow':
        args=contract['method_config']['batch']['args']
        native+=['feb12_train_ours.py','--dataset','imagenet_256','--datadir',data,
          '--model_type','DiT-XL/2','--image_size','256','--f','8','--num_levels','2',
          '--num_in_channels','4','--num_out_channels','4','--use_mask',
          '--time_seg_small',str(args['activations'][0]),'--label_dropout','0.1','--num_classes','1000',
          '--num_epoch','120' if entry['id'].endswith('_paper') else '1200',
          '--batch_size',str(batch),'--lr',str(c['optimizer']['lr']),'--use_mot','--use_ema',
          '--ema_decay',str(c['optimizer'].get('ema_decay')),'--no_lr_decay','--pretrained_autoencoder_ckpt',assets.get('eqvae','@eqvae@')]
        missing+=['Add exact paper path/stage flags; source epoch-only trainer lacks matched accumulation/max-update controls and best CFG is unidentified.']
    elif paper=='relay':
        native+=['train.py','--data',data,'--outdir','@original_output@','--cond=1','--arch=adm',
          '--batch='+str(batch),'--batch-gpu='+str(micro),'--lr='+str(c['optimizer']['lr']),
          '--precond='+('edm' if entry['id'].endswith('_low') else 'blur'),
          '--augment=0.12','--dropout=0.1','--fp16=1','--eff-attn=True',
          '--ls='+('25' if entry['id'].endswith('_low') else '1'),
          '--ema='+('50' if entry['id'].endswith('_low') else '0.5')]
        if not entry['id'].endswith('_low'):native+=['--up-scale=4','--block-scale=0.15','--prob-length=0.93','--blur-sigma-max=3.0']
        missing+=['Exact cascade allocation/image-clock/EMA/loss-scale/architecture flags and native ZIP label reader required.']
    elif paper=='ssd':
        native+=['scripts/image_train.py','dataset.name=celeba','dataset.data_dir='+data,
          'model.image_size=64','ssd.ssd_config_flag=true','ssd.num_levels=4','diffusion.predict_xstart=true',
          'training.batch_size='+str(batch),'training.microbatch='+str(micro),'training.lr='+str(c['optimizer']['lr'])]
        missing+=['Native Hydra train loop has no independent exact update cap; lr_anneal_steps would change LR.']
    elif paper=='mdm':
        native+=['ml-mdm-matryoshka/ml_mdm/clis/train_parallel.py','--config',
          str(source/'ml-mdm-matryoshka/configs/models'/('cc12m_1024x1024.yaml' if entry['id'].endswith('1024') else 'cc12m_256x256.yaml'))]
        missing+=['Original streaming/T5/vision curriculum/data/global batch must be recovered; CC12M is not the released Flickr training run.']
    elif paper=='rmd':
        native+=['-m','rmd.cli','train','--config',str(source/'configs/train_wan_1_3b.yaml'),
          '--data_path',data,'--output_dir','@original_output@']
        missing+=['Teacher/student/fake/FSDP/local pretrained assets and paper14B main-table settings are not source-core scratch training.']
    else:
        native+=['src/train.py'];missing+=['Best released full pipeline/config is not identified.']
    missing.append('Full native CLI vs unified CLI data/RNG/DDP lifecycle has not been certified. No best-result guarantee.')
    unified=[sys.executable,'-m','umdiff.cli','train','--config',str(config),'--data',data,
      '--output','@unified_output@','--batch-size',str(batch),'--seed',str(c['seed']),'--device','cpu']
    if isinstance(updates,int):unified+=['--steps',str(updates)]
    return dict(original=dict(cwd=str(cwd),argv=native),unified=dict(cwd=str(ROOT),argv=unified),
      certified=False,blocked_reasons=missing,
      warning='Entrypoint references need the listed overlays/assets/lifecycle work; they are not ready matched launch commands.')
