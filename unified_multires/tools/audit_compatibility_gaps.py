#!/usr/bin/env python3
"""CPU counterexamples for uncovered author branches, not new parity claims.

No weights/data are downloaded. Author AST bodies execute unchanged with small,
explicit dependencies. Toy predictions isolate orchestration/loss differences;
they are NOT evidence for author-network or image-quality equivalence.
"""
import ast
import bisect
import datetime
import hashlib
import inspect
import json
import math
import os
import pprint
import sys
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / '_deps')]
import einops
import numpy as np
import torch
from umdiff import build, native
from umdiff.batches import Prepared, SSDBatch
from umdiff.engine import Method
from umdiff.randomness import RandomStream
from umdiff.regularizers import ScaleEquivariance
from umdiff.samplers import PPFlowSampler
from umdiff.ssd import ScaleSpaceSampler
from umdiff.targets import MSE

torch.set_num_threads(1)
torch.manual_seed(20261008)
OUT = ROOT / 'evidence' / 'compatibility_audit'
CHECKS = []


def difference(a, b):
    return float((a.detach().double() - b.detach().double()).abs().max())


def record(name, **values):
    CHECKS.append(dict(name=name, status='confirmed', **values))


def dct_weighting():
    fn = native.extract('dctdiff/sde.py', ['LSimple'],
                        {'torch': torch, 'ScoreModel': object})['LSimple']
    shape = (2, 4, 6)
    clean = torch.randn(shape)
    noise = torch.randn(shape)
    state = torch.randn(shape)
    parameter = torch.nn.Parameter(torch.linspace(-.4, .7, 6))
    score = SimpleNamespace(sde=SimpleNamespace(sample=lambda x: (torch.ones(2)*.3, noise, state)),
                            noise_pred=lambda x, t: x * parameter)
    weight = torch.tensor([.2, .5, .8, 1.1, 1.4, 2.])
    weighted = fn(score, clean, reweight=weight)
    unweighted = fn(score, clean, reweight=1.)
    unified = MSE()(score.noise_pred(state, None), noise)[0]
    grad_weighted = torch.autograd.grad(weighted.mean(), parameter, retain_graph=True)[0]
    grad_unified = torch.autograd.grad(unified.mean(), parameter)[0]
    assert torch.equal(unweighted, unified)
    assert difference(weighted, unified) > .01
    assert difference(grad_weighted, grad_unified) > .01
    record('dctdiff_frequency_reweight', native_weighted=weighted.tolist(), unified=unified.tolist(),
           loss_max_abs=difference(weighted, unified), gradient_max_abs=difference(grad_weighted, grad_unified),
           native_reweight_one_control_exact=True, element_weights=weight.tolist(),
           interpretation='Unmodified author LSimple, fixed tensors and trainable per-frequency toy prediction. Existing oracle uses reweight=1; nonuniform weights change the objective and gradients.')


class ToyPPF(torch.nn.Module):
    patch_level = 3
    t1 = .5
    t2 = .75

    def forward_wo_cfg(self, x, t, y):
        return torch.where(y[:, None, None, None] == 1000, -torch.ones_like(x), 2*torch.ones_like(x))


def ppflow_guidance():
    fn = native.method('ppflow/models.py', 'forward_with_cfg', {'torch': torch, 'bisect': bisect})
    net = ToyPPF()
    full = torch.zeros(2, 4, 4, 4)
    labels = torch.tensor([2, 1000])
    fields = []
    for t in (.25, .5, .5001, .75, .7501):
        output = fn(net, full, torch.full((2,), t), labels, [2., 3., 4.])
        # The fourth channel stays conditional/unconditional rather than guided.
        assert output[0, 3, 0, 0] == 2 and output[1, 3, 0, 0] == -1
        fields.append(dict(t=t, guided_first_channel=float(output[0, 0, 0, 0]), fourth_channel=float(output[0, 3, 0, 0])))
    actual = PPFlowSampler(steps=1)(net, full[:1], model_kwargs={'y': labels[:1]})
    expected = fn(net, full, torch.zeros(2), labels, [2., 3., 4.])[:1]
    assert difference(actual, expected) == 3.
    record('ppflow_stage_cfg_and_three_channel_mask', fields=fields, one_euler_step_max_abs=difference(actual, expected),
           interpretation='Released forward_with_cfg AST, toy conditional/unconditional field. Existing sampler always invokes forward_wo_cfg. Stage equality belongs to earlier stage via bisect_left; fourth latent channel must not receive generic all-channel CFG.')


def dimr_guidance():
    path = ROOT / 'sources/dimr/eval.py'
    raw = path.read_bytes()
    tree = ast.parse(raw)
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'cfg_nnet')
    class Toy(torch.nn.Module):
        def forward(self, x, log_snr, y):
            return [torch.where(y[:, None, None, None] == 1000, -torch.ones_like(x), 2*torch.ones_like(x))]
    x = torch.zeros(1, 4, 4, 4)
    times = torch.tensor([500.])
    values = []
    for scale, power in ((.4, -1), (3., 2.)):
        env = dict(torch=torch, math=math, nnet=Toy(), dataset=SimpleNamespace(K=1000),
                   device='cpu', cfg_scale=scale, cfg_scale_pow=power)
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
        guided = env['cfg_nnet'](x, times, torch.zeros(1), torch.tensor([2]))
        conditional = env['nnet'](x, torch.zeros(1), torch.tensor([2]))[-1]
        assert difference(guided, conditional) > .5
        values.append(dict(scale=scale, power=power, guided=float(guided.flatten()[0]),
                           conditional=float(conditional.flatten()[0]), max_abs=difference(guided, conditional)))
    native.AUDIT.append(dict(source='sources/dimr/eval.py', sha256=hashlib.sha256(raw).hexdigest(),
                              line_span=[node.lineno, node.end_lineno], execution='unchanged nested cfg_nnet body, explicit toy prediction'))
    record('dimr_constant_and_decaying_cfg', cases=values,
           interpretation='Unmodified eval CFG function. Default conditional-only DiMRDPMSampler lacks both guidance conventions.')


def dfm_guidance():
    module = native.load('dfm/dfm_utils/samplers_dfm.py')
    class Toy(torch.nn.Module):
        def forward(self, state, times, y, drop_mask):
            return {k:torch.where(y[:, None, None, None] == 1000, -torch.ones_like(v), 2*torch.ones_like(v))
                    for k, v in state.items()}, None
    initial = {0:torch.zeros(1, 1, 2, 2), 1:torch.zeros(1, 1, 4, 4)}
    labels = torch.tensor([2])
    method = Method({'clock': {'name': 'dfm', 'args': {'steps': [4, 4], 'thresholds': [.5]}}})
    unified = method.sample(Toy(), initial, model_kwargs={'y': labels})
    base = module.dfm_euler_sampler(Toy(), initial, labels, cfg_scale=1., num_steps_per_scale=[4,4], stage_thresholds=[.5])
    assert all(torch.equal(unified[k], base[k]) for k in initial)
    guided = module.dfm_euler_sampler(Toy(), initial, labels, cfg_scale=3., guidance_low=.1, guidance_high=.6,
                                     num_steps_per_scale=[4,4], stage_thresholds=[.5])
    errors = {str(k):difference(unified[k], guided[k]) for k in initial}
    assert max(errors.values()) > .1
    record('dfm_per_scale_guidance_window', no_cfg_control_exact=True, final_band_max_abs=errors,
           interpretation='Full released sampler vs actual unified product-state sampler, identical toy model and scheduler. CFG decisions depend on each band time, not a single global timestep.')


def mdm_mixed_batch():
    high = torch.nn.Parameter(torch.arange(1., 7.).reshape(6, 1, 1, 1))
    low = torch.nn.Parameter(torch.arange(2., 8.).reshape(6, 1, 1, 1))
    pred = [high, low]
    target = [torch.zeros_like(x) for x in pred]
    subject = SimpleNamespace(_config=SimpleNamespace(multi_res_weights=None, use_double_loss=True),
                              mixed_ratio=np.array([2/3, 1.]), loss_fn=torch.nn.MSELoss(reduction='none'))
    env = native.statements('mdm/ml-mdm-matryoshka/ml_mdm/diffusion.py', 367, 386,
                            {'self': subject, 'pred': pred, 'tgt': target, 'x_t': pred})
    native_loss = env['loss']
    actual = Prepared(pred, target, torch.zeros(6)).loss(pred)[0]
    ng = torch.autograd.grad(native_loss.mean(), (high, low), retain_graph=True)
    ug = torch.autograd.grad(actual.mean(), (high, low))
    assert torch.count_nonzero(ng[0][-2:]) == 0 and torch.count_nonzero(ug[0][-2:]) == 2
    record('mdm_mixed_resolution_batch_mask', native_per_example=native_loss.tolist(), unified_per_example=actual.tolist(),
           loss_max_abs=difference(native_loss, actual), high_gradient_max_abs=difference(ng[0], ug[0]),
           excluded_native_gradients_zero=True,
           interpretation='Unchanged released loss loop, ratio 2:1 -> cumulative fractions [2/3,1]. Separate source NestedModel also crops high-resolution computation; a static per-head scalar cannot express this example mask.')


def ssd_rescaled_time():
    fn = native.method('ssd/guided_diffusion/scale_space_diffusion.py', '_scale_timesteps')
    settings = SimpleNamespace(rescale_timesteps=True, num_timesteps=8)
    expected = fn(settings, torch.tensor([7]))
    betas = np.linspace(.001, .02, 8).tolist()
    images = torch.zeros(1, 1, 2, 2)
    batch = SSDBatch([2]*8, betas, {2:2}, rescale_timesteps=True)
    train_time = batch(images, torch.tensor([7]), RandomStream(1)).time
    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.times=[]
        def forward(self, state, time, cond_rs=None):
            self.times.append(time.clone()); return torch.zeros_like(state)
    net = Spy()
    out = ScaleSpaceSampler([2]*8, betas, {2:2}, lanczos_iters=4, power_iters=2)(net, images, RandomStream(1))
    assert torch.isfinite(out).all() and torch.equal(train_time, expected)
    assert float(net.times[0][0]) == 7 and float(expected[0]) == 875
    record('ssd_train_sample_time_rescaling', author_and_unified_training_time=float(expected[0]),
           unified_sampler_first_time=float(net.times[0][0]), training_control_exact=True,
           interpretation='rescale_timesteps=True is supported by unified batch and author, but not exposed/applied by unified sampler. Default False remains unaffected.')


class EasyDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    @staticmethod
    def init_recursively(x): return EasyDict(x)


def diffusability_random_scale():
    fn = native.method('diffusability/src/training/loss.py', 'maybe_compute_scale_equiv_reg',
                       {'torch':torch, 'np':np, 'einops':einops, 'F':torch.nn.functional, 'EasyDict':EasyDict})
    scales = [.5, .25]
    settings = SimpleNamespace(weight=.3, scale_factors=scales, resample_strategy='mean',
                               temporal_scale_factors=[], ignore_lpips=True,
                               cut_dct2d_high_freqs=SimpleNamespace(max_cut_ratio=0.))
    subject = SimpleNamespace(cfg=SimpleNamespace(scale_equiv_reg=settings))
    # Explicit reconstruction dependency is MSE on both sides; no LPIPS model.
    subject.compute_rec_losses = lambda a,b,**kw: EasyDict(rec=(a-b).square().flatten(1).mean(1))
    latent = torch.randn(2, 1, 2, 8, 8)
    images = latent*.5
    decode = lambda x: x.square()
    native_decoder = lambda *a,latents,**kw: decode(latents)
    np.random.seed(42)
    state = np.random.get_state()
    chosen = float(np.random.choice(scales))
    np.random.set_state(state)
    expected = fn(subject, native_decoder, EasyDict(total=torch.zeros(2)), latent, images, None, None).total
    first = ScaleEquivariance(factor=2, mode='mean', weight=.3)(decode, latent, images)
    second = ScaleEquivariance(factor=4, mode='mean', weight=.3)(decode, latent, images)
    control = first if chosen == .5 else second
    assert torch.equal(expected, control)
    assert difference(expected, first+second) > .001
    record('diffusability_random_scale_vs_sum', selected_scale=chosen, native=expected.tolist(),
           unified_two_regularizer_sum=(first+second).tolist(), loss_max_abs=difference(expected, first+second),
           selected_single_scale_control_exact=True,
           interpretation='Unchanged author random-choice regularizer body with explicitly shared MSE dependency. Configuring two current regularizers runs both every step, which differs from random 2x/4x selection.')


def dimr_label_drop():
    # Source wrapper is small; AST extraction avoids dataset/download imports.
    env = native.extract('dimr/datasets.py', ['CFGDataset'], {'random':SimpleNamespace(random=lambda:0.),
                         'Dataset':torch.utils.data.Dataset})
    wrapped = env['CFGDataset']([(torch.zeros(4,2,2), 3)], .1, 1000)
    _, native_label = wrapped[0]
    from umdiff.cli import conditions
    spec = {'conditioning': {'y': {'kind': 'labels', 'classes': 1000}}}
    actual, _ = conditions(spec, {'labels':torch.tensor([3])}, 'cpu', 1)
    assert native_label == 1000 and int(actual['y'][0]) == 3
    rejected = False
    try: conditions(spec, {'labels':torch.tensor([1000])}, 'cpu', 1)
    except ValueError: rejected = True
    assert rejected
    record('dimr_cfg_training_label_dropout', native_dropped_label=native_label, unified_label=int(actual['y'][0]),
           pre_dropped_null_label_rejected_by_preset=True,
           interpretation='Author dataset RNG forced into the drop branch; existing generic labels conditioner has no dropout and its classes=1000 preset rejects null token1000. A custom field conditioner can transport preprocessed labels, but is not automatic recipe parity.')


def ssd_same_resolution_many_times():
    fn = native.method('ssd/guided_diffusion/resample.py', 'sample',
                       {'np':np, 'th':torch, 'pprint':pprint, 'logger':SimpleNamespace(debug=lambda *x:None)})
    resolutions = [4]*4 + [2]*4
    subject = SimpleNamespace(weights=lambda:np.ones(8), t_res_dict=dict(enumerate(resolutions)))
    np.random.seed(2)
    times, weights, _ = fn(subject, 8, 'cpu',
                         {'t_res_sampling_mode':'one_res_many_t_except_one_t_for_res_change'}, resolutions)
    assert times.unique().numel() > 1
    assert all(resolutions[int(t)] == resolutions[max(int(t)-1, 0)] == 4 for t in times)
    rejected = False
    try:
        SSDBatch(resolutions, np.linspace(.001,.02,8).tolist(), {4:4,2:2})(torch.zeros(8,1,4,4), times, RandomStream(2))
    except ValueError as e:
        rejected = str(e) == 'Native SSD uses one timestep/resolution for the whole batch'
    assert rejected
    record('ssd_default_same_resolution_many_times', native_times=times.tolist(),
           native_importance_weights=weights.tolist(), unified_batch_rejected=True,
           interpretation='Actual released sampler default permits distinct times with the same input/output resolution. Unified SSDBatch rejects this valid branch. Same-t uniform sampling can retain the uniform marginal objective, but changes batch correlation and training variance.')


def rdd_respaced_network_time():
    path = ROOT.parent/'diffusion/respace.py'
    raw = path.read_bytes()
    tree = ast.parse(raw)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_WrappedModel')
    env = {'th':torch}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), env)
    class Spy(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.times=[]
        def forward(self, x, time):
            self.times.append(time.clone()); return torch.zeros_like(x)
    mapping = [2,5,9,15]
    initial = torch.zeros(1,1,4,4)
    spy = Spy()
    env['_WrappedModel'](spy, mapping, 16)(initial, torch.tensor([3]))
    expected = int(spy.times[0][0])
    from umdiff.gaussian_objective import GaussianSampler
    spy = Spy()
    GaussianSampler(path={'betas':np.linspace(.001,.02,4).tolist(), 'timestep_map':mapping})(spy, initial, RandomStream(2))
    actual = int(spy.times[0][0])
    assert expected == 15 and actual == 3
    native.AUDIT.append(dict(source=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                              symbols=['_WrappedModel'], execution='unchanged migrated RDD wrapper class'))
    record('rdd_respaced_network_time', native_first_network_time=expected, unified_first_network_time=actual,
           accepted_timestep_map=mapping,
           interpretation='Path timestep_map is accepted and used for drift coefficients, but the current GaussianSampler calls the network with compressed indices. Migrated SpacedDiffusion wraps the model and maps back to original time. Existing identity-map tests remain valid.')


def real_dctdiff_weighting():
    model = build('model', {'name':'native_dctdiff', 'args':dict(embed_dim=32, depth=2, num_heads=4,
                                                            tokens=16, low_freqs=4, num_classes=-1)}).eval()
    state = torch.randn(2,16,24)
    noise = torch.randn_like(state)
    time = torch.tensor([.3,.7])
    prediction = model(state, time*999)
    score = SimpleNamespace(sde=SimpleNamespace(sample=lambda _: (time,noise,state)),
                            noise_pred=lambda *a,**k: prediction)
    fn = native.extract('dctdiff/sde.py', ['LSimple'], {'torch':torch, 'ScoreModel':object})['LSimple']
    weighted = fn(score, state, reweight=torch.linspace(.1,1.9,24))
    actual = MSE()(prediction,noise)[0]
    control = fn(score,state,reweight=1.)
    params=list(model.parameters())
    a=torch.autograd.grad(weighted.mean(),params,retain_graph=True,allow_unused=True)
    b=torch.autograd.grad(actual.mean(),params,allow_unused=True)
    maximum=max(difference(x,y) for x,y in zip(a,b) if x is not None)
    assert torch.equal(control,actual) and maximum>1e-5 and difference(weighted,actual)>1e-4
    record('dctdiff_actual_uvit_weighted_gradient', backbone=type(model).__module__+'.'+type(model).__name__,
           parameters=sum(p.numel() for p in params), native_weighted=weighted.tolist(),unified=actual.tolist(),
           loss_max_abs=difference(weighted,actual),parameter_gradient_max_abs=maximum,
           unweighted_control_exact=True,
           interpretation='Actual reduced author UViT, common synthetic state/time/noise and forward graph. Author LSimple weighting versus current unified metric, all parameter gradients compared. Random weights, not a paper-quality model.')


def real_ppflow_guidance():
    from umdiff.native_models import activate
    activate('ppflow',['models','transport','modules'])
    from transport.transport import Transport,ModelType,PathType,WeightType,Sampler
    objective=Transport(model_type=ModelType.VELOCITY,path_type=PathType.LINEAR,
                        loss_type=WeightType.NONE,train_eps=0.,sample_eps=0.)
    sample=Sampler(objective).sample_ode(sampling_method='euler',num_steps=13)
    results=[]
    for levels in (2,3):
        model=build('model',{'name':'native_ppflow','args':dict(input_size=16,patch_size=2,patch_level=levels,
                         in_channels=4,hidden_size=32,depth=2,num_heads=4,class_dropout_prob=.1,learn_sigma=True)}).eval()
        with torch.no_grad():
            for name,p in model.named_parameters():
                if p.ndim>=2 and not torch.any(p) and ('final' in name or 'proj_out' in name or 'out' in name):
                    p.normal_(std=.01)
        initial=torch.randn(2,4,16,16)
        labels=torch.tensor([0,1])
        with torch.no_grad():
            control=sample(initial,model.forward_wo_cfg,y=labels)[-1]
            actual=PPFlowSampler(steps=12)(model,initial,model_kwargs={'y':labels})
            expected=sample(torch.cat([initial,initial]),model.forward_with_cfg,
                            y=torch.cat([labels,torch.tensor([1000,1000])]),cfg_scales=[2.,3.,4.][:levels])[-1][:2]
        assert torch.equal(actual,control) and difference(actual,expected)>1e-5
        results.append(dict(levels=levels,no_cfg_full_trajectory_endpoint_exact=True,
                            guided_final_latent_max_abs=difference(actual,expected)))
    record('ppflow_actual_backbone_guided_sampling',cases=results,
           interpretation='Actual reduced author PPFlow 2/3stage networks and original transport Euler sampler. Same synthetic initial, labels, nonzero random final heads and 12 intervals. Unguided control exact, guided final latent differs. No training/weights/data download.')


def main():
    functions = [dct_weighting, ppflow_guidance, dimr_guidance, dfm_guidance,
                 mdm_mixed_batch, ssd_rescaled_time, diffusability_random_scale, dimr_label_drop,
                 ssd_same_resolution_many_times, rdd_respaced_network_time,
                 real_dctdiff_weighting, real_ppflow_guidance]
    for fn in functions:
        fn()
        print(fn.__name__ + ': confirmed', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    result = dict(audited_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  status='passed_expected_counterexamples', execution_device='cpu', gpu_used=False,
                  scope='Synthetic CPU branch counterexamples: 10 isolated branch probes with explicit toy dependencies, plus actual reduced author UViT gradients and PPFlow 2/3stage guided inference. No full-recipe or quality claim.',
                  checks=CHECKS, native_execution=native.AUDIT,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'PROBES.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status':result['status'], 'counterexamples':len(CHECKS), 'output':str(OUT/'PROBES.json')}))


if __name__ == '__main__': main()
