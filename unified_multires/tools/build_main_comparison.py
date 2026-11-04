"""Build auditable, fully expanded common-DiT experiment configurations."""
import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

from umdiff.rdd_native import diffusion_backend
from umdiff.experiment import validate_math

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'configs/comparison_imagenet256'
ANCHOR = ROOT/'configs/paired_experiments/rdd_unified_xl_300_300_400.json'
old = json.loads(ANCHOR.read_text())
ARCH = copy.deepcopy(old['model']['args'])
ARCH.update(num_classes=1000)
TRAINING = copy.deepcopy(old['training'])
TRAINING['input_transform']['args']['model']['args']['path'] = '@sd_vae_ema@'
TRAINING['input_transform_scope'] = 'microbatch'
TRAINING['optimizer'].update(optimizer_name='adamw', eps=1e-8, precision='float32',
                             grad_clip=None, lr_schedule=None, ema_update_every=1,
                             ema_warmup=False, ema_style='add_alpha')
RUN = dict(updates=400000, global_batch=256, micro_batch=32, world_size=1,
           seed=0, save_every=5000, log_every=100, allow_tf32=True)
ASSETS = dict(dataset='ImageNet-1K original train; 256 ADM center crop + random flip',
              data_env='IMAGENET_TRAIN_DIR', vae_env='SD_VAE_EMA_DIR',
              vae_id='stabilityai/sd-vae-ft-ema', posterior='sample_each_visit',
              latent_scale=.18215, required_model_domain=[4,32,32],
              precomputed_sampled_latents_are_not_the_legacy_recipe=True)
GLOBAL_CHANGES = [
    'Common ImageNet-256 class conditioning, SD-VAE-ft-ema, shared DiTMultiRes-XL/2 backbone, AdamW and 400K updates.',
    'Equal data exposure is fixed; equal updates are NOT equal FLOPs. Record training compute/time and compare inference quality versus measured cost.',
    'This frontend uses one process, micro32 and accumulation8 for global256. Historical RDD used 8 ranks x32; RNG/data order and FP summation are not claimed identical.',
    'Main files are preregistered starting candidates, not already tuned best-performing configurations.',
]
rows = []


def model(time_scale=1000., **kwargs):
    return {'name':'comparison_dit', 'args':{'architecture':copy.deepcopy(ARCH), 'time_scale':time_scale, **kwargs}}


def base(name):
    c = dict(name=name, image_shape=[4,32,32], model=model(), training=copy.deepcopy(TRAINING),
             evidence={'claim':'common_backbone_controlled_transplant', 'synthetic_training_allowed':False,
                       'not_a_paper_best_result_recipe':True, 'no_asset_downloads':True})
    c['training']['time_sampler']={'name':'uniform'}
    return c


def allocate(total, widths):
    counts=[max(1,int(total*w/sum(widths))) for w in widths]
    while sum(counts)<total:
        j=max(range(len(counts)),key=lambda i:total*widths[i]/sum(widths)-counts[i]);counts[j]+=1
    while sum(counts)>total:
        j=max(range(len(counts)),key=lambda i:counts[i]-total*widths[i]/sum(widths));counts[j]-=1
    return counts


def stage(name, grids, args):
    c=base(name)
    c.update(task='multires',grids=[[r,r] for r in grids],first_frame=False,
             process={'name':name,'args':args})
    c['training']['time_sampler']={'name':'none'}
    return c


def save(identifier, c, source, changes, sampling, suite='main', topology=None):
    c=copy.deepcopy(c)
    c['name']=identifier
    # Version3 rule selected by the user: exactly RDD's sum of stage means.
    # Keep author/native version2 recipes and the RDD anchor untouched.
    aggregation=None
    if c.get('task')=='multires':
        count=len(c['grids'])
        c['process']['args']['stage_weights']=[1.]*count
        c['training_stage_reduction']='sum_stage_means'
        aggregation={'name':'rdd_sum_stage_means','stages':count,
                     'outer_weights':[1.]*count,'time_policy':'Keep each process conditional clock',
                     'noise_policy':'Existing declared process noise coupling retained; this change concerns aggregation only'}
    elif c.get('batch',{}).get('name') in ('ssd','dfm','lapflow'):
        name=c['batch']['name'];args=c['batch']['args']
        if name=='ssd':
            table=args['resolutions'];grids=sorted(set(table))
            samplers=[{'name':'discrete_subset','args':{
                'indices':[i for i,r in enumerate(table) if r==grid],
                'steps':len(table),'one_for_batch':True}} for grid in grids]
            time_policy='One shared minibatch time, uniform conditional on current resolution; original transition indices retained'
        else:
            settings=copy.deepcopy(c['training']['time_sampler']['args'])
            samplers=[{'name':name,'args':{**settings,'stage':i}} for i in range(args['levels'])]
            time_policy='Enumerate native active modes; keep conditional joint clocks, masks, band weights and noise'
        c['training_aggregation']={'name':'sum_stage_means','samplers':samplers}
        c['training']['time_sampler']={'name':'none'}
        aggregation={'name':'rdd_sum_stage_means','stages':len(samplers),
                     'outer_weights':[1.]*len(samplers),'time_policy':time_policy,
                     'noise_policy':'Fresh native noise draw per active mode; within-mode band coupling retained'}
    # Author spatial position is an allowed method-specific adapter. The
    # immutable RDD anchor and methods without transferable author PE keep
    # their existing explicit common-backbone choice.
    position = None
    for prefix, policy, scope in [
        ('pixelflow', 'pixelflow_rope', 'Author integer token coordinates and Diffusers0.32 2D RoPE; evaluated on the new latent patch2 grid.'),
        ('dfm', 'dfm_sincos', 'Released SiT fixed width-first 2D sin-cos. Applied to each common token grid; author same-grid band fusion is replaced by joint tokens, so token correspondence is a declared architecture transplant.'),
        ('lapflow', 'lapflow_sincos', 'Selected released use_rope=False configuration: separate fixed width-first 2D sin-cos per band. Does not select the optional RoPE ablation.'),
        ('pyramid_flow', 'pyramid_flow_rope', 'Author FLUX axis RoPE with T=1 and no history: local integer spatial coordinates. Head64 axes16/24/24 become head72 axes16/28/28 to fit the common DiT; this dimension adaptation is not an author configuration.'),
        ('pyramidal_wan_path', 'wan_rope', 'Official Wan axis RoPE restricted to T=1; allocation follows Wan at the common head dimension. PyramidalWan author code is unavailable; this is a declared Wan-backbone position choice, not PyramidalWan source parity.'),
    ]:
        if identifier.startswith(prefix+'_'):
            c['model']['args']['position']=policy
            position={'policy':policy,'scope':scope}
            break
    spec=dict(format='umdiff_experiment_v1',id=identifier,suite=suite,method=c,
              run=copy.deepcopy(RUN),assets=copy.deepcopy(ASSETS),sampling=sampling,
              comparison={'source':source,'changes':changes,'common_contract':GLOBAL_CHANGES,
                          'topology_coarse_to_fine':topology,
                          'selection':'Use validation data and an equal tuning-compute budget; do not select on final test scores.',
                          'evaluation':'Primary CFG=1; 50K generated RGB, same ADM reference and preprocessing. Metrics are a separate evaluation step.',
                          'runtime_scope':'CPU/single visible GPU CLI; existing device guard is retained; no GPU authorization in this file.'})
    if position is not None:
        spec['comparison']['position']=position
    if aggregation is not None:
        spec['comparison']['training_aggregation']=aggregation
        spec['comparison']['changes']=list(changes)+[
            'User-selected RDD rule: visit every training phase for each image; sum phase mean losses with coefficient1, no 1/K or stage-probability factors; one optimizer/EMA update after microbatch accumulation. This intentionally changes the author outer training objective/estimator.']
    validate_math(spec)
    dest=OUT/suite/(identifier+'.json');dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(spec,indent=2,ensure_ascii=False)+'\n')
    rows.append({'id':identifier,'suite':suite,'file':str(dest.relative_to(OUT)),
                 'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    return spec


# Preserve the legacy RDD algorithm and actual 400K/DDPM-250 sampling anchor.
rdd=copy.deepcopy(old)
rdd['training']=copy.deepcopy(TRAINING)
rdd['sampler']['args'].update(steps=[90,80,80],solver='ddpm',guidance_scale=1.,clip_clean=False)
rdd['evidence'].update(claim='legacy_rdd_algorithm_anchor',
                       unchanged=['model architecture','training_stages','training_stage_reduction','optimizer mathematical hyperparameters'],
                       sampling_source='multiscale_diffusion/2d/exp/imagenet256_unifiedXL300300400_400000_908080step_ddpm_cfg115.jsonl')
for key in ('model','training_stages','training_stage_reduction'):
    assert rdd[key]==old[key]
save('rdd_ddpm_xl_legacy',rdd,'local legacy RDD Unified-XL/2 400K',
     ['No RDD model, path, objective, stage lengths, prediction/variance semantics or optimizer hyperparameter change.',
      'Sampler restored from the actual legacy DDPM-250 experiment: coarse90/middle80/fine80, CFG1 primary. CFG1.5 is a separate legacy operating point.',
      'Single-process execution topology differs from the historical 8-rank launch; no bitwise historical trajectory claim.'],
     {'kind':'ddpm_jump','steps_coarse_to_fine':[90,80,80],'cfg':1.,'legacy_also_report_cfg':1.5},topology=[8,16,32])

ddpm=copy.deepcopy(rdd)
backend=diffusion_backend()
d=backend.create_diffusion_with_drift('',diffusion_steps=1000,c=None,T_c=0,T_c_plus_1=1000)
ddpm['training_stages']=[{'input':{'name':'identity'},'time_sampler':{'name':'discrete','args':{'steps':1000}},
                         'method':{'batch':{'name':'rdd_ddpm','args':{'path':{'betas':d.betas.tolist(),'timestep_map':d.timestep_map},'learned_variance':True}}}}]
ddpm['sampler']['args'].update(resolutions=[32],stage_lengths=[1000],steps=[250],solver='ancestral')
ddpm['evidence']={'claim':'fixed_resolution_ddpm_on_common_legacy_rope_dit','synthetic_training_allowed':False}
save('ddpm_xl_fixed',ddpm,'DDPM/DiT baseline on common backbone',
     ['Ordinary full-resolution DDPM, learned variance; uses common RDD RoPE backbone rather than original fixed sin-cos DiT.'],
     {'kind':'ancestral_respaced','steps':250,'cfg':1.},topology=[32])

flow=base('flow')
flow.update(path={'name':'flow','args':{'orientation':'data_to_noise'}},
            target={'name':'joint','args':{'quantity':'velocity'}},
            clock={'name':'linear','args':{'start':1.,'end':0.}})
save('flow_xl_fixed',flow,'ordinary linear flow baseline',
     ['Full-resolution velocity flow on the identical DiT core; no additional teacher or pretraining.'],
     {'kind':'generic_euler','sample_cli_steps':250,'cfg':1.},topology=[32])

fdm_changes=[
    'ADM UNet replaced by the common DiT; RGB DS replaced by average-pooled SD-VAE latents.',
    'Retain epsilon plus residual heads, within-stage signal drift, VP noise rescaling, P2 loss and stochastic boundary noise.',
    'All stage means are summed with unit coefficients under the common RDD rule; interval-length objective weights are removed. Shared image/noise pyramid remains a declared estimator-correlation change.',
    'Paper cosine formula has an endpoint inconsistency. The normalized-cosine candidate explicitly assumes a missing pi/2; linear candidate uses the unambiguous alternative from section3.1. Neither is claimed an exact author best recipe.'
]
for kind in ['cosine_assumption','linear']:
    bounds=[math.cos(j*math.pi/10) for j in range(6)] if kind=='cosine_assumption' else [1-j/5 for j in range(6)]
    bounds[-1]=0.
    widths=[a-b for a,b in zip(bounds,bounds[1:])]
    c=stage('fdm',[2,4,8,16,32],{'steps':allocate(250,widths),'bounds':bounds,'stage_weights':widths,
                                    'rescaling':'vp','prediction':'epsilon','loss_weight':'p2','eta':1.,'signal_up':'linear','noise_lift':'average','terminal':.9999})
    c['model']=model(heads=['epsilon','residual'])
    save('fdm_xl_5_'+kind,c,'https://arxiv.org/html/2210.04955',fdm_changes,
         {'kind':'fdm_ancestral','steps_coarse_to_fine':c['process']['args']['steps'],'cfg':1.,'clock_status':kind},topology=[2,4,8,16,32])

pf=stage('pyramid_flow',[8,16,32],{'steps':[20,20,20],'stage_range':[0,1/3,2/3,1.],
                                 'train_steps':1000,'shift':1.,'network_clock':'author','boundary_policy':'native_psd'})
save('pyramid_flow_xl_3',pf,'https://arxiv.org/abs/2410.05954',
     ['Video DiT replaced by common image DiT; omit temporal pyramid/autoregressive history/text conditioning.',
      'Replace author VAE/latent scale with shared SD-VAE-ft-ema; both construct the spatial pyramid after VAE encoding.',
      'Retain spatial flow endpoints, correlated transition noise and author network-clock mapping.',
      'Explicit native_psd boundary uses a PSD square root for the author singular gamma=1/3 covariance; original Cholesky is undefined. Same noise law, coupled CPU draws.'],
     {'kind':'euler','steps_coarse_to_fine':[20,20,20],'cfg':1.},topology=[8,16,32])

pw=stage('pyramidal_wan_path',[8,16,32],{'steps':[20,20,10],'natural_bounds':[1.,.9412,.5858,0.]})
save('pyramidal_wan_path_xl_3',pw,'https://arxiv.org/html/2601.04792',
     ['Wan/video/pretrained fine-tuning replaced by common class-conditioned DiT trained from scratch on SD-VAE images.',
      'Retain natural/global clock conversion and orthogonal noise transition; omit temporal compression, first-frame special handling and teacher/DMD.',
      'Wan-derived natural thresholds are an explicit starting candidate, not re-estimated optimal SD-VAE thresholds; paper implementation, not author-code parity.'],
     {'kind':'euler','steps_coarse_to_fine':[20,20,10],'cfg':1.},topology=[8,16,32])

pixel=stage('pixelflow',[4,8,16,32],{'steps':[30,30,30,30],'train_steps':1000,'shift':1.,'gamma':-1/3,
                                  'solver':'native_dopri5','guidance_scale':1.,'atol':1e-6,'rtol':.001})
save('pixelflow_xl_4',pixel,'sources/pixelflow/configs/pixelflow_xl_c2i.yaml',
     ['RGB patch4 packed PixelFlow transformer replaced by common latent patch2 DiT with the author PixelFlow positional rule.',
      'Retain four-stage endpoint/clock and equicorrelated boundary noise; unpacked stage minibatches replace mixed-resolution packing.',
      'Use the released adaptive Dopri5 backend and tolerances; common CFG=1 retains the author doubled-state arithmetic. No author best-FID claim.'],
     {'kind':'native_dopri5','saved_points_coarse_to_fine':[30]*4,'atol':1e-6,'rtol':.001,'cfg':1.,
      'nfe':'adaptive; saved output points are not model evaluations; measure actual calls'},topology=[4,8,16,32])

def ssd_config(grids, resolutions=None):
    c=base('ssd');c['model']={'name':'ssd_grid','args':{'model':model(time_scale=1.)}}
    if resolutions is None:
        resolutions=[grids[len(grids)-1-min(int((1-math.sqrt(1-t/999))*len(grids)),len(grids)-1)] for t in range(1000)]
    common={'resolutions':resolutions,'betas':np.linspace(.0001,.02,1000).tolist(),
            'unet_res_layer_map':{str(g):g for g in grids},'rescale_timesteps':False}
    c['batch']={'name':'ssd','args':{**copy.deepcopy(common),'loss_weight':'min_snr_5'}}
    c['sampler']={'name':'ssd','args':{**copy.deepcopy(common),'clip_clean':False,'lanczos_iters':40,'power_iters':20}}
    c['training']['time_sampler']={'name':'discrete','args':{'steps':1000,'one_for_batch':True}}
    return c

ssd=ssd_config([4,8,16,32])
save('ssd_xl_4_convex',ssd,'sources/ssd/configs/{ssd,diffusion}.yaml',
     ['Flexi-UNet replaced by shared DiT; necessary output-grid resize is explicit; RGB process moved into SD-VAE latents.',
      'Retain linear1000 beta table, ConvexDecay0.5, bilinear operator posterior/Lanczos and min-SNR5 objective.',
      'Visit every resolution phase per update, with one common conditional time per phase minibatch; disable [-1,1] clean clipping because SD latents are not bounded RGB.'],
     {'kind':'operator_ddpm','steps':1000,'cfg':1.,'respacing':'not substituted with a different beta table'},topology=[4,8,16,32])

dfm=base('dfm');dfm['model']=model(channels=[4,4])
dfm['training']['time_sampler']={'name':'dfm','args':{'weights':[.9,.1],'weighting':'lognormal','curve':'linear'}}
dfm.update(batch={'name':'dfm','args':{'levels':2,'stage_weights':[.9,.1],'weighting':'lognormal','repa_weight':0.,'native_debug_rng':True}},
           representation={'name':'laplacian','args':{'levels':2,'up':'bilinear','order':'coarse_first'}},
           clock={'name':'dfm','args':{'steps':[30,10],'thresholds':[.1]}})
save('dfm_xl_2',dfm,'sources/dfm/experiments/{train,generation}/dfm_sit_b_256.yaml',
     ['Released two-scale SiT replaced by common RDD DiT blocks with joint tokens and per-band time modulation; REPA stays disabled as in the released training YAML (enc_type:none, proj_coeff:0).',
      'Retain Laplacian bands, lognormal active-band time and shifted-logit-normal earlier-band times, independent band noise and active-band loss sum. Enumerate both active modes with outer coefficients1/1 instead of NumPy stage selection0.9/0.1; author logging RNG draws remain but no longer choose the next phase.',
      'Use shared SD-VAE-ft-ema instead of released ft-mse and the common optimizer/execution policy; not an unpublished SE-FLUX/DiT or paper-best-score recipe.'],
     {'kind':'joint_band_euler','steps_per_phase':[30,10],'thresholds':[.1],'cfg':1.},topology=[16,32])

lap=base('lapflow');lap['model']=model(channels=[4,4],product_order='fine_first',causal_scales=True)
lap['training']['time_sampler']={'name':'lapflow','args':{'activations':[.5,0.]}}
lap.update(batch={'name':'lapflow','args':{'levels':2,'activations':[.5,0.]}},
           sampler={'name':'lapflow','args':{'levels':2,'activations':[.5,0.],'solver':'dopri5','step_size':.01,'atol':1e-5,'rtol':1e-5,'guidance_scale':1.}})
save('lapflow_xl_2',lap,'https://arxiv.org/html/2602.19461#S5.SS1',
     ['Replace MoT/expert parameters by shared RDD DiT blocks; retain joint band tokens, necessary per-band interface and causal attention mask.',
      'Use SD-VAE-ft-ema rather than EQVAE at256; retain delayed linear flow and paper T=.5.',
      'Native-compatible two-level sampler still computes its unused coarse tail; include those calls in measured cost.'],
     {'kind':'dopri5','activation_fine_first':[.5,0.],'atol':1e-5,'rtol':1e-5,'cfg':1.},topology=[16,32])

haar=stage('nonuniform_haar',[4,8,16,32],{'steps':[64]*4,'likelihood_weighting':False,'sampling':'sde'})
haar['model']=model(channels=[4,12,12,12])
save('nonuniform_haar_xl_4',haar,'https://arxiv.org/html/2207.09786#S4.SS1',
     ['Four separate UNet score networks replaced by a shared DiT core; explicit Haar coefficient channel projections are added.',
      'Retain Haar states, component-specific VP clocks and progressive active-band SDE; use the unweighted DSM variant.',
      'ImageNet128 pixel task moved to ImageNet256 SD-VAE latent task; positional encoding and estimator are controlled choices.'],
     {'kind':'euler_maruyama','steps_coarse_to_fine':[64]*4,'cfg':1.},topology=[4,8,16,32])

# Controlled scalar-state group: same grids, nominal 400/300/300 partition,
# same 90/80/80 sampling allocation. Boundary renormalization stays native.
control_note=['Controlled variant: grids8/16/32, nominal noise-to-data phase lengths400/300/300. RDD/fDM/Pyramid Flow/PyramidalWan/PixelFlow use90/80/80 calls; SSD explicitly retains1000.',
              'Clock labels are method-specific; this does NOT assert equal physical SNR, equal per-band clocks or equal FLOPs. Boundary rescaling is retained.']
save('rdd_ddpm_xl_legacy',rdd,'same immutable RDD anchor',control_note,
     {'kind':'ddpm_jump','steps_coarse_to_fine':[90,80,80],'cfg':1.},suite='controlled',topology=[8,16,32])
cf=stage('fdm',[8,16,32],{'steps':[90,80,80],'bounds':[1.,.6,.3,0.],'stage_weights':[.4,.3,.3],
                         'rescaling':'vp','prediction':'epsilon','loss_weight':'p2','eta':1.})
cf['model']=model(heads=['epsilon','residual'])
cp=copy.deepcopy(pf);cp['process']['args'].update(steps=[90,80,80],stage_range=[0.,.4,.7,1.])
cw=copy.deepcopy(pw);cw['process']['args'].update(steps=[90,80,80],natural_bounds=[1.,.6,.3,0.])
cx=stage('pixelflow',[8,16,32],{'steps':[90,80,80],'stage_range':[0.,.4,.7,1.],'train_steps':1000,'gamma':-1/3,
                             'solver':'euler','guidance_scale':1.})
for name,c in [('fdm',cf),('pyramid_flow',cp),('pyramidal_wan_path',cw),('pixelflow',cx)]:
    save(name+'_xl_shared_3',c,'controlled variant; see main method for original changes',control_note,
         {'kind':name,'steps_coarse_to_fine':[90,80,80],'cfg':1.},suite='controlled',topology=[8,16,32])
cs=ssd_config([8,16,32],[32]*300+[16]*300+[8]*400)
save('ssd_xl_shared_3',cs,'controlled SSD topology/partition',
     control_note+['SSD retains its exact1000-step operator sampler; equal250-call inference is not claimed or achieved by changing training betas.'],
     {'kind':'operator_ddpm','steps':1000,'cfg':1.,'matched_250_calls':False},suite='controlled',topology=[8,16,32])

manifest={'format':'umdiff_comparison_manifest_v1','entries':rows,
          'excluded':['relay','pddpm','mrflow','bottleneck','edify_full_cascade','diffusability','cat'],
          'separate_scope':{'speed':'Inference-only extension; not part of the scratch training comparison.',
                            'mdm_dimr_ppflow_nami':'Architecture-centered methods outside the selected fixed-backbone main comparison.',
                            'joint_band_controls':'DFM/LapFlow/Haar are main joint-band rows; not falsely included in the common scalar-clock controlled group.'},
          'rdd_reference':{'config':str(ANCHOR.relative_to(ROOT)), 'sha256':hashlib.sha256(ANCHOR.read_bytes()).hexdigest(),
                           'native_trainer':'multiscale_diffusion/2d/train_RDD_imagenet256_unified.py',
                           'legacy_world_size':8,'legacy_per_rank_batch':32,'global_batch':256},
          'full_training_started':False}
OUT.mkdir(parents=True,exist_ok=True)
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({'configs':len(rows),'main':sum(r['suite']=='main' for r in rows),'controlled':sum(r['suite']=='controlled' for r in rows),'output':str(OUT)}))
