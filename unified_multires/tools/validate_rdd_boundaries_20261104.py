"""CPU old/new RDD boundary and complete DDPM/DPM trajectory conformance.

The frozen pre-change old sampler and current old sampler are AST-extracted;
no copying of the production unified stage loop is used as the reference.
Original diffusion solver and DiT definitions are deliberately shared backends.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
sys.path[:0] = [str(ROOT), str(ROOT/'_deps')]
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
import torch
import torch.nn.functional as F
from umdiff import build
from umdiff.rdd_native import NativeRDDStages, diffusion_backend
from umdiff.randomness import GlobalRandomStream, RandomStream

BACKEND = diffusion_backend()
from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
from diffusion.stage_transition import lift_noise_with_nullspace_2d
from diffusion.fdm_rescaling import validate_fdm_rescaling, rescale_alpha_sigma, spatial_dimension_ratio

OLD = WORK/'2d/sample_fid_unified.py'
FROZEN = ROOT/'evidence/rdd_reproduction_20261104/prechange_sources/multiscale_diffusion/2d/sample_fid_unified.py'
VIDEO = WORK/'rdd_video/sample_fastwan_native_vsa.py'
VIDEO_FROZEN = ROOT/'evidence/rdd_reproduction_20261104/prechange_sources/multiscale_diffusion/rdd_video/sample_fastwan_native_vsa.py'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_sha(t):
    return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def clone(value):
    if torch.is_tensor(value): return value.detach().clone()
    if isinstance(value, dict): return {k: clone(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return type(value)(clone(v) for v in value)
    return value


def compare(a, b, name, checks):
    if torch.is_tensor(a):
        valid = a.shape == b.shape and a.dtype == b.dtype
        exact = valid and torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))
        err = float((a.double()-b.double()).abs().max()) if a.numel() and a.shape == b.shape else 0.
        checks.append(dict(name=name, shape=list(a.shape), dtype=str(a.dtype), byte_exact=bool(exact), max_abs=err))
        assert exact, (name, err)
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), name
        for key in a: compare(a[key], b[key], f'{name}/{key}', checks)
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b), (name, len(a), len(b))
        for i, (x, y) in enumerate(zip(a, b)): compare(x, y, f'{name}/{i}', checks)
    else:
        assert a == b, (name, a, b)


def functions(path, names, env):
    tree = ast.parse(path.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(selected) == len(names), (path, names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), env)
    return env


def sampler_env(path):
    names = {'_redraw_upsample', '_noise_filling_upsample', '_bilinear_upsample',
             '_posterior_upsample', '_fdm_rescaled_upsample_probe', '_do_upsample',
             '_run_stage', '_sample_batch_unified', '_sample_batch_multistage'}
    return functions(path, names, dict(torch=torch, F=F, _extract_into_tensor=_extract_into_tensor,
        lift_noise_with_nullspace_2d=lift_noise_with_nullspace_2d,
        validate_fdm_rescaling=validate_fdm_rescaling, rescale_alpha_sigma=rescale_alpha_sigma,
        spatial_dimension_ratio=spatial_dimension_ratio))


def record(net):
    saved = net.forward
    calls = []
    def forward(x, t, *args, **kwargs):
        out = saved(x, t, *args, **kwargs)
        calls.append(clone(dict(x=x, t=t, args=args, kwargs=kwargs, output=out)))
        return out
    net.forward = forward
    return calls


def make_model(multistage=False):
    if multistage:
        from umdiff.rdd_multistage import NativeRDDMultistage
        torch.manual_seed(3841)
        result = NativeRDDMultistage(models={str(r):{'name':'native_rdd','args':dict(
            input_size=r,hidden_size=w,depth=d,num_heads=4,patch_size=2,in_channels=4,
            num_classes=1000,class_dropout_prob=.1,learn_sigma=True)}
            for r,w,d in [(32,64,3),(16,48,2),(8,32,2)]})
        with torch.no_grad():
            for parameter in result.parameters():parameter.normal_(0,.03)
        return result.eval()
    torch.manual_seed(3841)
    model = build('model', {'name': 'native_rdd_unified', 'args': dict(
        hidden_size=32, depth=2, num_heads=4, patch_size=2, in_channels=4,
        num_classes=1000, class_dropout_prob=.1, learn_sigma=True, base_size=16)}).eval()
    with torch.no_grad():
        for parameter in model.parameters(): parameter.normal_(0, .03)
    return model


def make_diffusions(steps, resolutions=(32, 16, 8), lengths=(300, 300, 400)):
    limits = [0]
    for length in lengths: limits.append(limits[-1]+length)
    return [BACKEND.create_diffusion_with_drift(str(steps[-1-i]),
        diffusion_steps=limits[-1], c=i if i < len(resolutions)-1 else None,
        T_c=limits[i], T_c_plus_1=limits[i+1], sample_scale=resolutions[i]//resolutions[i+1] if i < len(resolutions)-1 else 2)
        for i in range(len(resolutions))]


def reference(path, model, solver, mode, seed, batch, cfg, steps, resolutions, lengths, options=None, multistage=False):
    env = sampler_env(path)
    trace = []
    run = env['_run_stage']
    upsample = env['_do_upsample']
    def run_record(*args, **kwargs):
        result = run(*args, **kwargs)
        trace.append(result['sample'][:batch].detach().clone())
        return result
    def up_record(*args, **kwargs):
        result = upsample(*args, **kwargs)
        trace.append(result[0][:batch].detach().clone())
        return result
    env['_run_stage'] = run_record
    env['_do_upsample'] = up_record
    torch.manual_seed(seed)
    if multistage:
        result=env['_sample_batch_multistage'](batch,list(reversed(make_diffusions(steps,resolutions,lengths))),
            [model.for_resolution(r) for r in reversed(resolutions)],list(reversed(resolutions)),
            1000,cfg>1,cfg,solver,mode,steps,torch.device('cpu'),torch.float32)
    else:
        opts=dict(options or {})
        if 'guidance_scale_per_stage' in opts:opts['cfg_scale_per_stage']=opts.pop('guidance_scale_per_stage')
        result = env['_sample_batch_unified'](
            batch, make_diffusions(steps, resolutions, lengths), resolutions, model, 1000, cfg > 1,
            cfg, solver, mode, steps, torch.device('cpu'), torch.float32, **opts)
    return result, trace, torch.get_rng_state().clone()


@torch.no_grad()
def decoded(vae, x):
    return vae.decode(x.float()/.18215).sample


@torch.no_grad()
def case(vae, solver, boundary, cfg, outdir, seed=641, batch=2, steps=None, resolutions=None, lengths=None, suffix="", options=None, multistage=False):
    steps = steps or [80, 80, 90]
    resolutions = resolutions or [32, 16, 8]
    lengths = lengths or [300, 300, 400]
    name = f'{solver}_{boundary}_cfg{cfg:g}{suffix}'
    options=options or {}
    checks = []
    report = dict(options=options,multistage=multistage,name=name, solver=solver, boundary=boundary, cfg=cfg,
        seed=seed, batch=batch, grids=list(reversed(resolutions)), steps_coarse_to_fine=steps,
        stage_lengths_fine_to_coarse=lengths, checks=checks)
    try:
        a = make_model(multistage); b = deepcopy(a)
        if multistage:
            traces_a={r:record(a.for_resolution(r)) for r in resolutions}
            traces_b={r:record(b.for_resolution(r)) for r in resolutions}
        else:
            calls_a = record(a); calls_b = record(b)
        source = FROZEN if boundary in ('fresh','bilinear','legacy_initial_noise') else OLD
        mode = {'fresh':'noise_filling','legacy_initial_noise':'innovation_preserving'}.get(boundary,boundary)
        expected, ref_trace, ref_rng = reference(source, a, solver, mode, seed, batch, cfg, steps, resolutions, lengths, options, multistage)
        # Match the public old evaluator: labels, coarsest conditional noise,
        # then optional unconditional half. NativeRDDStages creates the latter.
        torch.manual_seed(seed)
        labels = torch.randint(0, 1000, (batch,))
        initial = torch.randn(batch, 4, resolutions[-1], resolutions[-1])
        native = NativeRDDStages(resolutions=resolutions, stage_lengths=lengths, steps=steps, solver=solver, boundary=boundary, guidance_scale=cfg, **options)
        actual, trace = native(b, initial, GlobalRandomStream(), {'y': labels}, True)
        if multistage:
            calls_a=[item for r in reversed(resolutions) for item in traces_a[r]]
            calls_b=[item for r in reversed(resolutions) for item in traces_b[r]]
        compare(calls_a, calls_b, 'all_model_calls', checks)
        compare(ref_trace, trace, 'stage_endpoints_and_boundaries', checks)
        compare(expected, actual, 'final_latent', checks)
        compare(ref_rng, torch.get_rng_state(), 'RNG', checks)
        compare(decoded(vae, expected), decoded(vae, actual), 'decoded_RGB_float', checks)
        assert torch.isfinite(actual).all()
        # Full global/private-stream replay; no extra model trace capture.
        torch.manual_seed(seed)
        torch.randint(0, 1000, (batch,)); torch.randn(batch, 4, resolutions[-1], resolutions[-1])
        before = torch.get_rng_state().clone()
        stream = RandomStream(seed)
        stream.generator.set_state(before)
        calls_b.clear()
        alias = {'fresh':'refresh','noise_filling':'coupled','legacy_initial_noise':'isometric'}.get(boundary,boundary)
        replay = NativeRDDStages(resolutions=resolutions, stage_lengths=lengths, steps=steps, solver=solver, boundary=alias, guidance_scale=cfg, **options)
        alias_result, alias_trace = replay(b, initial, stream, {'y': labels}, True)
        compare(actual, alias_result, 'alias_private_stream_final', checks)
        compare(trace, alias_trace, 'alias_private_stream_trace', checks)
        compare(before, torch.get_rng_state(), 'private_stream_preserves_outer_rng', checks)
        compare(ref_rng, stream.generator.get_state(), 'private_stream_rng_consumption', checks)
        report.update(status='passed', model_calls=len(calls_a), compared_tensor_pairs=len(checks),
            max_abs=max(c['max_abs'] for c in checks), reference=str(source.relative_to(WORK)),
            final_sha256=tensor_sha(actual), final_float_rgb_sha256=tensor_sha(decoded(vae, actual)),
            trace_sha256=[tensor_sha(t) for t in trace],
            scope='Full old evaluator stage assembly and current NativeRDDStages; original DiT and diffusion solver backends reused. All model calls, stage boundary/endpoint tensors, final float VAE pixels, RNG. Nonzero random small DiT, random tiny VAE; no trained-quality claim.')
        print(json.dumps({k:report[k] for k in ('name', 'status', 'model_calls', 'compared_tensor_pairs', 'max_abs')}), flush=True)
    except Exception:
        report.update(status='failed', error=traceback.format_exc())
        print(report['error'], flush=True)
    (outdir/(name+'.json')).write_text(json.dumps(report, indent=2)+'\n')
    return report


def video_boundary_function(path):
    env = functions(path, {'upsample_nested', 'upsample_nested_torch_nearest', 'lift_noise_with_nullspace'},
                    dict(torch=torch, math=math))
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sample_three_call_dmd')
    # Extract the literal production boundary dispatch, including all branch
    # arithmetic; GPU event/model/timing infrastructure is deliberately outside.
    dispatch = next(n for n in ast.walk(fn) if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Compare)
                    and isinstance(n.test.left, ast.Name) and n.test.left.id == 'transition_mode')
    params = 'clean previous_state previous_sigma sigma_next shape generator device transition_mode boundary_executor'.split()
    extracted = ast.FunctionDef(name='boundary',
        args=ast.arguments(posonlyargs=[], args=[ast.arg(arg=s) for s in params], vararg=None,
                           kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[]),
        body=[deepcopy(dispatch), ast.Return(ast.Name('state', ast.Load()))], decorator_list=[])
    exec(compile(ast.fix_missing_locations(ast.Module([extracted], type_ignores=[])), str(path), 'exec'), env)
    return env['boundary']


def video_cases():
    old = video_boundary_function(VIDEO_FROZEN); new = video_boundary_function(VIDEO)
    results = []
    for dtype in (torch.float32, torch.bfloat16):
        for shape in ((3, 8, 8), (6, 8, 8)):
            torch.manual_seed(191)
            clean = torch.randn(2, 4, 3, 4, 4).to(dtype)
            noisy = torch.randn_like(clean)
            args = (clean, noisy, torch.tensor(.6, dtype=dtype), torch.tensor(.4, dtype=dtype), shape)
            for before, after in [('noise_filling', 'fresh'), ('noise_filling', 'refresh'),
                                  ('innovation_preserving', 'noise_filling')]:
                g1 = torch.Generator().manual_seed(191); g2 = torch.Generator().manual_seed(191)
                a = old(*args, g1, torch.device('cpu'), before, 'native')
                b = new(*args, g2, torch.device('cpu'), after, 'native')
                checks = []
                compare(a, b, 'boundary', checks); compare(g1.get_state(), g2.get_state(), 'RNG', checks)
                results.append(dict(dtype=str(dtype), target_grid=list(shape), old_mode=before,
                    canonical_mode=after, status='passed', checks=checks,
                    scope='AST-extracted original native boundary dispatch only; no Wan forward, GPU kernels or full video sampler execution'))
    return results


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path,
        default=ROOT/'evidence/rdd_reproduction_20261104/ddpm_boundary')
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    from diffusers import AutoencoderKL
    torch.manual_seed(510)
    vae = AutoencoderKL(in_channels=3, out_channels=3, latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*4, up_block_types=('UpDecoderBlock2D',)*4,
        block_out_channels=(8, 8, 8, 8), layers_per_block=1,
        norm_num_groups=4, sample_size=256).eval()
    start = time.time()
    files = [Path(__file__), OLD, FROZEN, VIDEO, VIDEO_FROZEN,
             ROOT/'umdiff/rdd_native.py', ROOT/'umdiff/rdd_multistage.py', ROOT/'umdiff/boundaries.py',
             WORK/'diffusion/gaussian_diffusion_with_drift.py',
             WORK/'DiT/models_multires.py']
    hashes = {str(p.relative_to(WORK)):sha(p) for p in files}
    results = []
    for solver in ('ddpm', 'dpm1', 'dpm2'):
        for cfg in (1., 1.5):
            for boundary in ('fresh', 'noise_filling'):
                results.append(case(vae, solver, boundary, cfg, args.output))
    negatives = []
    for solver in ('ddpm', 'dpm1', 'dpm2'):
        for cfg in (1., 1.5):
            pair = [r for r in results if r['solver'] == solver and r['cfg'] == cfg]
            if all(r['status'] == 'passed' for r in pair):
                identical_coarse = pair[0]['trace_sha256'][0] == pair[1]['trace_sha256'][0]
                boundary_differs = pair[0]['trace_sha256'][1] != pair[1]['trace_sha256'][1]
                final_differs = pair[0]['final_sha256'] != pair[1]['final_sha256']
                assert identical_coarse and boundary_differs and final_differs
                negatives.append(dict(solver=solver, cfg=cfg, identical_coarse_stage=True,
                    different_first_boundary=True, different_final=True))
    extended_recipes = [
        dict(suffix='_allocation400_300_300', solver='ddpm', cfg=1., lengths=[400,300,300]),
        dict(suffix='_allocation100_300_600', solver='ddpm', cfg=1., lengths=[100,300,600]),
        dict(suffix='_allocation600_300_100', solver='ddpm', cfg=1., lengths=[600,300,100]),
        dict(suffix='_two_stage16_32', solver='ddpm', cfg=1., resolutions=[32,16], lengths=[500,500], steps=[125,125]),
        dict(suffix='_two_stage8_32', solver='ddpm', cfg=1., resolutions=[32,8], lengths=[500,500], steps=[125,125]),
        dict(suffix='_historical244NFE', solver='dpm2', cfg=1.5, steps=[40,40,45]),
        dict(suffix='_budget35', solver='ddpm', cfg=1., steps=[10,10,15]),
        dict(suffix='_proportional150_75_25', solver='ddpm', cfg=1., steps=[150,75,25], lengths=[100,300,600]),
    ]
    for recipe in extended_recipes:
        for boundary in ('fresh','noise_filling'):
            results.append(case(vae, boundary=boundary, outdir=args.output, **recipe))
    extra_recipes=[
        dict(suffix='_bilinear',solver='ddpm',cfg=1.5,boundary='bilinear'),
        dict(suffix='_bilinear',solver='dpm1',cfg=1.5,boundary='bilinear'),
        dict(suffix='_legacy_initial',solver='ddpm',cfg=1.,boundary='legacy_initial_noise'),
        dict(suffix='_legacy_initial_sigma',solver='ddpm',cfg=1.5,boundary='legacy_initial_noise',options={'sigma_h':.2}),
        dict(suffix='_sigma',solver='ddpm',cfg=1.5,boundary='noise_filling',options={'sigma_h':.2}),
        dict(suffix='_drift',solver='ddpm',cfg=1.5,boundary='fresh',options={'drift_scale':.5}),
        dict(suffix='_stagecfg',solver='ddpm',cfg=1.1,boundary='fresh',options={'guidance_scale_per_stage':[1.1,1.3,1.5]}),
        dict(suffix='_multistage',solver='ddpm',cfg=1.5,boundary='fresh',multistage=True),
        dict(suffix='_multistage',solver='ddpm',cfg=1.5,boundary='noise_filling',multistage=True),
        dict(suffix='_multistage',solver='ddpm',cfg=1.,boundary='bilinear',multistage=True),
        dict(suffix='_multistage',solver='dpm1',cfg=1.5,boundary='bilinear',multistage=True),
        dict(suffix='_multistage',solver='dpm2',cfg=1.5,boundary='fresh',multistage=True,steps=[40,40,45]),
    ]
    for recipe in extra_recipes:
        results.append(case(vae,outdir=args.output,**recipe))
    try:
        videos = video_cases(); video_error = None
    except Exception:
        videos = []; video_error = traceback.format_exc()
    report = dict(status='passed' if all(r['status']=='passed' for r in results) and not video_error else 'failed',
        device='cpu', cuda_initialized=torch.cuda.is_initialized(), elapsed_seconds=time.time()-start,
        cases=[{k:v for k,v in r.items() if k != 'checks'} for r in results],
        tensor_checks=sum(len(r['checks']) for r in results), negative_controls=negatives,
        video_boundary_checks=videos, video_error=video_error,
        source_sha256_before=hashes, source_sha256_after={str(p.relative_to(WORK)):sha(p) for p in files})
    report['source_stable_during_test'] = report['source_sha256_before'] == report['source_sha256_after']
    (args.output/'SUMMARY.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('status','tensor_checks','elapsed_seconds','cuda_initialized','source_stable_during_test')}))
    if report['status'] != 'passed': raise SystemExit(1)


if __name__ == '__main__': main()
