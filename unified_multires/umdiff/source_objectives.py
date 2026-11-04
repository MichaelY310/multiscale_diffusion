"""Pinned author objectives alongside unified objectives for controlled comparisons.

Uses the same supplied minibatch, author network and externally chosen lifecycle.
This is an author-source core adapter, not an unmodified native CLI trainer.
"""
from types import SimpleNamespace
import random
import numpy as np
import torch
import torch.nn.functional as F
from . import build, native
from .engine import Method
from .native_models import activate
from .randomness import GlobalRandomStream
STREAM=GlobalRandomStream()

def unpack_model(spec):
    return spec['args']['model'] if spec['name'] == 'adapter' else spec

def setup(name, config, images=None, labels=None, input_conditions=None):
    """Return separate native/unified forward functions, input and scope notes."""
    method = Method(config)
    batch = 8 if name.startswith('ppflow') else 4 if name == 'pixelflow' else 3 if name == 'mdm' else 1
    fixture=images is None
    if fixture:images = torch.randn(batch, *config['image_shape'])
    else:batch=len(images)
    conditional=any(v.get('kind')=='labels' for v in config.get('training',{}).get('conditioning',{}).values())
    if labels is None:
        if conditional and not fixture:raise ValueError('Conditional author/unified training requires real labels')
        labels=torch.arange(batch,device=images.device) if fixture else torch.zeros(batch,device=images.device,dtype=torch.long)
    input_conditions=input_conditions or {}
    clock = build('time_sampler', config['training']['time_sampler'])
    kwargs = {'y': labels}
    extra = {}
    notes = ['Synthetic inputs and source random initialization; no pretrained assets.',
             'Full released network dimensions and resolution retained; short run and smaller logical batch.',
             'CPU attention/autocast kernels; this is not CUDA/DDP kernel certification.']
    if name.startswith('dctdiff'):
        source = native.load('dctdiff/sde.py')
        netcfg = unpack_model(config['model'])['args']
        # Real input grid, codec and source EBFR values from the selected config.
        images = images.clamp(-1, 1)
        clean = method.representation.encode(images)
        statistics = config['metric']['args']['dct_statistics']
        block = statistics['block']
        from umdiff.dctcodec import zigzag
        env = native.statements('dctdiff/train.py', 78, 97, dict(
            config=SimpleNamespace(dataset=SimpleNamespace(**statistics, block_sz=block,
                                                           low_freqs=statistics['frequencies'])),
            low2high_order=np.array(zigzag(block)), np=np, torch=torch, device='cpu',
            logging=SimpleNamespace(info=lambda *a: None)))
        kwargs = {'y': labels} if netcfg['num_classes'] > 0 else {}
        def original(net, step):
            return source.LSimple(source.ScoreModel(net, 'noise_pred', source.VPSDE(
                SNR_scale=config['path']['args']['snr_scale'])), clean,
                reweight=env['reweight_by_std'], **kwargs)
    elif name == 'dimr':
        author = native.extract('dimr/train.py', ['check_zip', 'get_skip', 'stp', 'mos', 'Schedule', 'LSimple'],
                                dict(np=np, torch=torch, multi_scale_targets=native.load('dimr/sde.py').multi_scale_targets))
        schedule = author['Schedule'](np.array(config['batch']['args']['betas']))
        def original(net, step):
            # Author CFGDataset dropout precedes time/noise generation.
            y = torch.tensor([1000 if random.random() < .1 else int(v) for v in labels])
            return author['LSimple'](images, lambda x, log_snr, **kw: net(x, log_snr, **kw),
                                     schedule, config['batch']['args']['loss_weights'], y=y)
    elif name.startswith('ppflow'):
        activate('ppflow', ['models', 'transport', 'modules'])
        from transport.transport import Transport, ModelType, PathType, WeightType
        objective = Transport(model_type=ModelType.VELOCITY, path_type=PathType.LINEAR,
                              loss_type=WeightType.NONE, train_eps=0., sample_eps=0.)
        def original(net, step):
            return objective.training_losses(net, images, config['batch']['args']['levels'], kwargs)['loss']
        notes.append('SiT pretrained initialization unavailable; constructor initialization retained on both sides.')
    elif name == 'pixelflow':
        activate('pixelflow', ['pixelflow'])
        import pixelflow.data_in1k as data
        from diffusers.models.embeddings import _get_2d_rotary_pos_embed_np
        from pixelflow.scheduling_pixelflow import PixelFlowScheduler
        data.get_2d_rotary_pos_embed = _get_2d_rotary_pos_embed_np
        netcfg = unpack_model(config['model'])['args']
        cfg = SimpleNamespace(model=SimpleNamespace(params=SimpleNamespace(**netcfg)),
                              scheduler=SimpleNamespace(num_stages=4, num_train_timesteps=1000))
        scheduler = PixelFlowScheduler(1000, 4)
        kwargs = {}
        def original(net, step):
            packed = data.collate_fn(list(zip(images, labels.tolist())), cfg, scheduler)
            prediction = net(packed['pixel_values'], packed['timesteps'],
                             class_labels=torch.tensor(packed['input_ids']), latent_size=packed['batch_latent_size'],
                             pos_embed=packed['pos_embed'], cu_seqlens_q=packed['cumsum_q_len'], cu_seqlens_k=None,
                             seqlen_list_q=packed['seqlen_list_q'], seqlen_list_k=None)
            return native.statements('pixelflow/train.py', 120, 126,
                                     dict(model_output=prediction, target=packed['target_values'], batch=packed, torch=torch))['loss']
    elif name in ('dfm','dfm_yaml','dfm_repa'):
        activate('dfm', ['models', 'dfm_utils', 'loss'])
        import loss as author
        objective = author.DFMSILoss(stage_weights=[.9, .1],weighting=config['batch'].get('args',{}).get('weighting','uniform'))
        teacher_features=input_conditions.get('teacher_features')
        if name=='dfm_repa' and teacher_features is None:raise ValueError('REPA requires explicitly supplied local teacher_features')
        def original(net, step):
            # Source's random debug logging consumes extra NumPy RNG after its
            # stage draw. Couple the draw explicitly and restore the exact state
            # advanced by the actual source loss, including debug logging.
            if name=='dfm_repa':
                loss,projection=objective(net,images,model_kwargs=kwargs,zs=teacher_features)
                return loss.mean()+projection.mean()*config['batch']['args']['repa_weight']
            return objective(lambda x, t, **kw: (net(x, t, **kw), None), images, model_kwargs=kwargs)[0]
        notes.append('Released SiT architecture differs from paper architecture; optional native debug NumPy draws enabled in the formal recipe to preserve subsequent stage choices.')
    elif name in ('lapflow','lapflow_paper'):
        activate('lapflow', ['models', 'sampler'])
        from sampler.ours_sample import generate_xt_and_u
        args = SimpleNamespace(noise_var_factor=None, path_type='linear', use_gvp=False,
                               epsilon=None, variance_aware=False, use_mot=True,
                               loss_weight_scale_0=1., loss_weight_scale_1=1., num_levels=2)
        def original(net, step):
            t = torch.rand(batch)
            stage = random.randint(0, 1)
            activation = config['batch']['args'].get('activations',[.4,0.])[stage]
            t = activation + t * (1 - activation)
            target, state, weights, variance = generate_xt_and_u([.33, .67], config['batch']['args'].get('activations',[.4,0.])[0], images,
                                                                  t[:, None, None, None], stage, 2, False, args)
            class NativeCall:
                def __call__(self,time,x,**kw):
                    # Source squeezes the time batch for B=1. The controlled
                    # micro1 frontend must restore its batch dimension.
                    return net(x,time.reshape(batch),**kw)
            return native.statements('lapflow/feb12_train_ours.py',415,427,
                dict(training_model=NativeCall(),t=t[:,None,None,None],x_t=state,u=target,
                     loss_scales=weights,var_scale=variance,y=labels,args=args,
                     scale_selector=stage,F=F,epoch=0))['loss']
        notes.append('Paper ImageNet hyperparameters with released architecture; EQVAE unavailable.' if name=='lapflow_paper' else 'Best run not identified in public repo; this is released XL/2-level CLI component profile.')
    elif name == 'ssd':
        activate('ssd', ['guided_diffusion'])
        from guided_diffusion import scale_space_diffusion as author, resample
        author.dist = torch.distributed if torch.distributed.is_initialized() else SimpleNamespace(get_rank=lambda: 0, barrier=lambda: None)
        a = config['batch']['args']
        mapping = {int(k): v for k, v in a['unet_res_layer_map'].items()}
        settings = dict(resolution_schedule='ConvexDecay_0.5',
                        t_res_sampling_mode='one_res_many_t_except_one_t_for_res_change',
                        multires_training_mode='flexi_unet', unet_res_layer_map=mapping, scale_min=8)
        source = author.ScaleSpaceDiffusion(betas=np.array(a['betas']), model_mean_type=author.ModelMeanType.START_X_RS,
                                            model_var_type=author.ModelVarType.FIXED_SMALL, loss_type=author.LossType.MSE,
                                            ssd_config=settings, resolutions_array=[8, 16, 32, 64],
                                            mse_loss_weight_type='min_snr_5', rescale_timesteps=False)
        for op in source.ops.values():
            op.H_out, op.W_out = op.size_out
        times = resample.UniformSampler(source)
        times.prepare_sampling(settings, [8, 16, 32, 64])
        kwargs = {}
        def original(net, step):
            t, weight, _ = times.sample(batch, 'cpu', settings, [8, 16, 32, 64])
            return source.training_losses(net, images, t, None)['loss'] * weight
    elif name in ('mdm','mdm1024'):
        from .native_models import import_mdm
        import_mdm()
        from ml_mdm import diffusion as author, samplers
        cfg = author.NestedDiffusionConfig(sampler_config=samplers.SamplerConfig(
            num_diffusion_steps=1000, prediction_type=samplers.PredictionType.V_PREDICTION,
            loss_target_type=samplers.PredictionType.DDPM, schedule_type=samplers.ScheduleType.DEEPFLOYD,
            schedule_shifted=True,schedule_shifted_power=2 if name=='mdm1024' else 1), use_double_loss=True, no_use_residual=True,
            use_vdm_loss_weights=False, mixed_ratio=None if name=='mdm1024' else '2:1',
            multi_res_weights='16:4:1' if name=='mdm1024' else None)
        text = input_conditions.get('lm_outputs')
        if text is None:
            spec=unpack_model(config['model'])['args']['config']
            dimension=2048
            while True:
                candidate=spec.get('conditioning_feature_dim',-1)
                if candidate>0:dimension=candidate
                if 'inner_config' not in spec:break
                spec=spec['inner_config']
            text = torch.randn(batch, 8, dimension)
        mask = input_conditions.get('lm_mask')
        if mask is None:mask = torch.ones(batch, 8)
        kwargs = dict(conditioning=text, cond_mask=mask, micros={})
        def original(net, step):
            return author.NestedDiffusion(net, cfg).get_loss(
                dict(images=images, lm_outputs=text, lm_mask=mask))[0]
        notes.append('Synthetic T5-shaped embeddings; no vision/inner initialization or full progressive curriculum.')
    elif name == 'rdd':
        from umdiff.rdd_native import diffusion_backend
        backend = diffusion_backend()
        limits = [0, 300, 600, 1000]
        diffusions = [backend.create_diffusion_with_drift('', diffusion_steps=1000, c=i if i < 2 else None,
                      T_c=limits[i], T_c_plus_1=limits[i + 1], sample_scale=2) for i in range(3)]
        def original(net, step):
            loss = 0.
            for i, diffusion in enumerate(diffusions):
                r = 2**i
                x = images if i == 0 else images.detach().reshape(batch, 4, 32//r, r, 32//r, r).mean((3, 5))
                t = torch.randint(0, diffusion.num_timesteps, (batch,))
                loss = loss + diffusion.training_losses(net, x, t, kwargs)['loss'].mean()
            return loss
    elif name.startswith('relay'):
        import scipy.stats as stats
        blur = native.load('relay/blurring.py')
        env = dict(torch=torch, np=np, st=stats, persistence=SimpleNamespace(persistent_class=lambda cls: cls),
                   block_noise=blur.block_noise, dct_2d=blur.dct_2d, idct_2d=blur.idct_2d)
        helpers = native.extract('relay/training/loss.py', ['DCTBlur', 'BlurLoss', 'EDMLoss'], env)
        objective = helpers['EDMLoss']() if name == 'relay_low' else helpers['BlurLoss'](
            up_scale=4, block_scale=.15, prob_length=.93, blur_sigma_max=3.)
        from .batches import relay_augmentation
        augment=relay_augmentation(config['batch'].get('args',{}).get('augmentation'))
        kwargs = {'class_labels': F.one_hot(labels, 1000).float()}
        def original(net, step):
            loss=objective(net, images, labels=kwargs['class_labels'],augment_pipe=augment)
            return loss if config['batch'].get('args',{}).get('native_loss',False) else loss.flatten(1).mean(1)
        notes.append('Single logical batch, no 8-rank averaging/accumulation; source loss scaling adjusted to actual batch for this conformance run.')
    else:
        raise ValueError(name)

    def port(net, step):
        nonlocal kwargs
        if name == 'dimr':
            kwargs = {'y': torch.tensor([1000 if random.random() < .1 else int(v) for v in labels])}
        if name == 'pixelflow':
            indices = (list(range(4)) * (batch//4 + 1))[:batch]
            random.shuffle(indices)
            context = dict(stage_indices=torch.tensor(indices), class_labels=labels)
            t = torch.randint(0, 1000, (batch,))
        else:
            t, context = clock(batch, 'cpu', STREAM)
        if name=='dfm_repa':context['teacher_features']=teacher_features
        return method.training(net, images, t, STREAM, model_kwargs=kwargs, context=context).loss
    return method, original, port, images, notes
