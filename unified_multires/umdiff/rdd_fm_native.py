"""Checkpoint-compatible RDD-FM image recipe.

The generic RDDProcess remains a configurable mathematical transplant. This
method preserves the old image trainer's stage order, independent Gaussian
draws, interval epsilon, all-stage SUM and the original Euler/midpoint backend.
"""
from copy import deepcopy
import importlib
import torch
from .engine import TrainingResult
from .rdd_native import diffusion_backend


class NativeRDDFM:
    def __init__(self, config):
        allowed = {'task', 'name', 'image_shape', 'model', 'training', 'process', 'evidence'}
        if set(config)-allowed:
            raise ValueError(f'Unknown native FM fields: {set(config)-allowed}')
        self.config = deepcopy(config)
        args = dict(config.get('process', {}))
        allowed_args = {'resolutions', 'bounds', 'steps', 'solver', 'boundary', 'cfg_scale', 'train_eps'}
        if set(args)-allowed_args:
            raise ValueError(f'Unknown native FM process fields: {set(args)-allowed_args}')
        self.resolutions = list(args.get('resolutions', [32, 16, 8]))  # fine -> coarse
        self.bounds = list(args.get('bounds', [0., .3, .6, 1.]))  # noise -> data
        self.steps = list(args.get('steps', [80, 80, 90]))  # coarse -> fine
        self.solver = args.get('solver', 'euler')
        self.boundary = args.get('boundary', 'noise_filling')
        self.boundary = {'refresh': 'fresh', 'nearest': 'fresh', 'redraw': 'fresh', 'coupled': 'noise_filling'}.get(self.boundary, self.boundary)
        self.cfg = float(args.get('cfg_scale', 1.))
        if self.boundary not in ('fresh', 'noise_filling', 'bilinear'):
            raise ValueError('Native FM boundary must be fresh (refresh), noise_filling or bilinear')
        if self.solver not in ('euler', 'midpoint', 'rk2'):
            raise ValueError('Native FM solver must be Euler or midpoint/RK2; FM dpm2 is not DPM-Solver')
        n = len(self.resolutions)
        if n < 1 or len(self.bounds) != n+1 or len(self.steps) != n or min(self.steps) < 1:
            raise ValueError('One interval and positive sampling step count per resolution required')
        if self.bounds[0] != 0 or self.bounds[-1] != 1 or any(a >= b for a, b in zip(self.bounds, self.bounds[1:])):
            raise ValueError('FM bounds must strictly ascend 0 -> 1')
        if any(a <= b or a % b for a, b in zip(self.resolutions, self.resolutions[1:])):
            raise ValueError('Native FM grids must be nested integers, finest first')
        if config['image_shape'] != [4, self.resolutions[0], self.resolutions[0]]:
            raise ValueError('Legacy image FM expects four-channel square VAE latents')
        if config.get('training', {}).get('time_sampler', {'name': 'none'}) != {'name': 'none'}:
            raise ValueError('Native FM draws a separate time inside each stage; configure time_sampler=none')
        diffusion_backend()
        self.backend = importlib.import_module('diffusion.flow_matching_with_drift')
        self.flow_matchers = []
        for i in range(n):
            j = n-1-i
            self.flow_matchers.append(self.backend.FlowMatchingWithDrift(
                c=i if i < n-1 else None,
                sample_scale=self.resolutions[i]//self.resolutions[i+1] if i < n-1 else 2,
                s_k=self.bounds[j], s_k_plus_1=self.bounds[j+1],
                train_eps=float(args.get('train_eps', 1e-5))))
        self.sampler = self
        self.last_sample_metrics = None

    def initial_shape(self):
        return [4, self.resolutions[-1], self.resolutions[-1]]

    def training(self, model, images, t, stream, noise=None, model_kwargs=None, active=None, context=None):
        if t is not None or noise is not None or active is not None:
            raise ValueError('Native FM owns independent per-stage time/noise draws')
        if list(images.shape[1:]) != self.config['image_shape']:
            raise ValueError('Latent shape does not match the native FM recipe')
        predictions, targets, states, terms = [], [], [], {}
        total = 0.0
        with stream.native_rng(images.device):
            for i, (res, fm) in enumerate(zip(self.resolutions, self.flow_matchers)):
                x = images
                if res != self.resolutions[0]:
                    f = self.resolutions[0]//res
                    x = images.detach().view(len(images), 4, res, f, res, f).mean(dim=(3, 5))
                # Preserve the exact old RNG and arithmetic ordering, including
                # model label dropout after this stage's t and epsilon draws.
                lo, hi = fm.s_k+fm.train_eps, fm.s_k_plus_1-fm.train_eps
                time = torch.rand(len(x), device=x.device)*(hi-lo)+lo
                eps = torch.randn_like(x)
                drift = fm.compute_drift(x) if fm.c is not None else None
                state = fm.q_sample(x, time, noise=eps, d=drift)
                target = fm.compute_velocity_target(x, eps, time, d=drift)
                pred = model(state, time*1000.0, **(model_kwargs or {}))
                if pred.shape[1] == x.shape[1]*2:
                    pred, _ = pred.chunk(2, dim=1)
                loss = (pred-target).square().mean(dim=list(range(1, pred.ndim)))
                total = total+loss.mean()
                predictions.append(pred); targets.append(target); states.append(state)
                terms[f'stage{i}/loss'] = loss
                terms[f'stage{i}/time'] = time
        return TrainingResult(total, predictions, targets, states, terms)

    @torch.no_grad()
    def sample(self, model, initial, steps=20, stream=None, model_kwargs=None, return_trajectory=False):
        if stream is None or list(initial.shape[1:]) != self.initial_shape():
            raise ValueError('Explicit stream and correctly shaped initial noise required')
        state = initial
        trace = [state.clone()] if return_trajectory else []
        nfes = []
        with stream.native_rng(initial.device):
            for j, fm in enumerate(reversed(self.flow_matchers)):
                if j:
                    res = self.resolutions[-1-j]
                    if self.boundary == 'noise_filling':
                        state = self.backend.noise_filling_transition_fm(state, clean, fm.s_k, res)
                    else:
                        state = self.backend.upsample_with_noise_fm(clean, fm.s_k, res,
                            mode='bilinear' if self.boundary == 'bilinear' else 'nearest')
                    if return_trajectory: trace.append(state.clone())
                out = fm.sample_ode(model, state, self.steps[j], model_kwargs=model_kwargs,
                                    cfg_scale=self.cfg, solver=self.solver)
                state, clean = out['sample'], out['pred_x1']
                nfes.append(out['nfe'])
                if return_trajectory: trace.append(state.clone())
        self.last_sample_metrics = {'nfe': sum(nfes), 'stage_nfe': nfes,
                                    'trace_scope': 'initial, stage boundaries and endpoints; model hooks expose every solver evaluation'}
        return (clean, trace) if return_trajectory else clean
