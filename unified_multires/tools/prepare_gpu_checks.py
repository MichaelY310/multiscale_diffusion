#!/usr/bin/env python3
"""Prepare/audit future CUDA checks without importing Torch or starting jobs.

This is an executable CPU planner, not a CUDA parity runner. Current device
authorization is CPU only. Unfinished native/CUDA adapters stay explicit.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from paired_experiments import compile_plan, load_manifest

OUT = ROOT / 'configs/gpu_checks'
EVIDENCE = ROOT / 'evidence/gpu_preparation_20261010'
BASELINE = ROOT / 'evidence/paired_experiments_20261010/SUMMARY.json'
INFERENCE_ONLY = {'speed', 'nami', 'mrflow'}

# The topology here describes source lifecycle tests, not the one-process
# fair-RDD recipe and not a request to allocate these resources.
SPECIAL = {
    'relay': ('Native FP16 loss scaling, nan/inf gradient repair, two-network cascade, image-count LR/EMA clocks.',
              '8-rank native reference; topology/layout must be confirmed for the selected best run.',
              ['Exact low/high checkpoint pair', 'Author ZIP data/labels and sampler row']),
    'ssd': ('Flexi-UNet scale/time sampling, original DDPM loop, constant LR independent of stop cap.',
            '2-rank released train_ssd.sh reference; best-row topology not uniquely identified.',
            ['Selected scale-count checkpoint', 'Exact source data transform and evaluation row']),
    'pixelflow': ('Packed variable-length CUDA attention, four-stage CFG/noise, Euler and native Dopri5; compare both CFG halves.',
                  'Native DDP; best-run batch/budget/topology unknown.',
                  ['Best XL checkpoint', 'Exact public YAML overlay and selected solver/CFG row']),
    'dfm': ('Native uniform/lognormal/REPA branches and continuous NumPy/Torch RNG, Accelerate accumulation and clipping.',
            'Native Accelerate distributed recipe; paper DiT/SE-FLUX branch differs from public SiT/SD-VAE.',
            ['Selected SiT initialization/SD VAE', 'Real REPA teacher/features when enabled', 'Paper SE-FLUX codec/DiT assets are a separate missing release']),
    'lapflow': ('MOT stage sampling, two/three levels, codec posterior sampling, native sampler time/path flags.',
                'Native multi-rank recipe; dataset/epoch/CFG selection must precede comparison.',
                ['Exact EQVAE/SD VAE and trained checkpoint', 'Selected stage-count/CFG row']),
    'ppflow': ('Two/three-level patch packing, SiT initialization, Heun CFG and packing-divisible microbatches.',
               'Native DDP; accumulation/max-update/local-VAE overlays remain unimplemented.',
               ['SiT-XL initialization', 'Exact SD VAE', 'Selected trained PPFlow checkpoint']),
    'dctdiff': ('FP16 Accelerator/scaler, UViT attention, dataset-specific DCT statistics/EBFR, DCT decode pixels.',
                'Native Accelerator lifecycle; selected per-dataset topology must be bound.',
                ['Original DCT/EBFR statistics', 'Selected UViT checkpoint', 'Original preprocessed dataset layout']),
    'mdm': ('Nested pyramid, per-YAML precision (256 fp16=0;1024 fp16=1), online Flan-T5, curriculum/mixed-resolution stream, activation recomputation.',
            'Native distributed streaming training; best global batch/topology unknown.',
            ['Exact Flan-T5-XL/tokenizer', 'Required inner-model initialization', 'Selected released MDM checkpoint and matching data stream']),
    'dimr': ('FP16 Accelerator, three-branch loss, moments/labels loader and Python CFG dropout; native DPM guidance semantics.',
             '16 ranks / 2 nodes in release; never emulate with the one-process fair recipe.',
             ['Original SD VAE', 'Author .npy posterior moments/labels', 'Selected 800-epoch release checkpoint']),
    'diffusability': ('SE2/SE4 codec and VGG branches, frozen/trainable codec, downstream generator training; optional GAN/temporal remain deferred.',
                     'Best full training lifecycle/batch/topology not released.',
                     ['Exact codec/VGG', 'Downstream pretrained generator/teacher', 'Complete best training recipe is unavailable']),
    'rmd': ('Video BCTHW, teacher/fake/student, transition1/2, warmup15, two optimizers, AMP/FSDP and optional GAN.',
            'Released Wan14B: 8 ranks, micro4, accumulation2 -> global64; paper batch4 is a distinct experiment.',
            ['Exact teacher/fake/student checkpoints and local text/VAE assets', 'Exact prompt set (image-free prompt-only distillation)', 'Paper14B/GAN configuration and source attribution remain unconfirmed']),
    'rdd': ('Native decoupled/joint targets, latent32/16/8, stage300/300/400; actual SD-VAE decode and optimizer/EMA clocks.',
            'Historical 8 ranks x micro32 -> global256; fair recipe 1 x16 x16 is a separate controlled run.',
            ['Exact SD VAE ema', 'Selected L/XL checkpoint', 'Real ImageNet labels and ADM transform']),
    'speed': ('Native DCT/Haar/FFT schedules and CFG/NFE; FLUX transformer plus real VAE pixels.',
              'Public inference only; no training/DDP parity claim.',
              ['Exact FLUX-dev transformer/text encoders/tokenizers/VAE', 'Fixed real prompts and benchmark row']),
    'nami': ('Trained progressive FLUX transformer, learned bridge and vendored Diffusers; real VAE pixels.',
             'Public inference only; paper training batch768 has no released trainer.',
             ['Exact trained NAMI transformer/bridge', 'Matching local text encoders/tokenizers/VAE', 'Fixed real prompts']),
    'mrflow': ('Native VAE -> PIL uint8 -> RRDB tiled SR -> resize -> posterior -> refinement, CFG and 12+1 NFE.',
               'Public inference only; FLUX and Qwen rows require separate models.',
               ['Exact FLUX/Qwen pipeline and text assets', 'Author-compatible RealESRGAN/RRDB', 'Fixed real prompts and SR/tile/sigma row']),
}

CATEGORIES = {
    'cuda_core_fp32': dict(
        scope='Same device/kernel, paired source objective vs unified objective; module activations, activation/parameter gradients, loss and continuous RNG.',
        updates=2, minimum_gpus=1,
        prerequisites=['Implement explicit device propagation; source_objectives.py still has CPU-only tensors/extracted contexts.',
                       'Move BOTH source and port inputs/labels/masks/noise to the same device; capture CPU+all visible CUDA+NumPy+Python RNG.',
                       'No CPU fallback or CPU attention shim may certify a CUDA kernel.']),
    'cuda_native_precision': dict(
        scope='Original dtype/autocast/master parameters/scaler; unscale, clip/repair, optimizer, EMA and overflow/skip behavior.',
        updates=2, minimum_gpus=1,
        prerequisites=['A GPU adapter for both independent optimizer lifecycles, including native precision policy.',
                       'Compare native dtype; BF16/FP16 cannot inherit FP32CPU certification.']),
    'cuda_sample_decode': dict(
        scope='Every stage boundary/CFG branch/noise draw, final latent and floating decoded pixels before rounding; source solver row and NFE.',
        updates=0, minimum_gpus=1,
        prerequisites=['GPU sampler/codec adapters and identical local model/tokenizer/VAE/SR assets.',
                       'Fix real prompt/labels, initial noise, posterior RNG, solver tolerances and preprocessing.']),
    'cuda_native_lifecycle': dict(
        scope='Actual author CLI vs unified CLI: accumulation/all-reduce, image/update clocks, checkpoint, fresh-process resume, data/worker RNG.',
        updates=2, minimum_gpus=2,
        prerequisites=['Complete/certify the author/unified native CLI pair on CPU first; current production_entrypoints are references only.',
                       'A 2-rank check is only a distributed smoke; the best-run topology requires a separate reference run.',
                       'NCCL/FSDP/multi-device behavior cannot be certified on one GPU or by CPU Gloo.']),
    'cuda_cost': dict(
        scope='CUDA-synchronized latency, actual forward calls/NFE, allocated/reserved peak memory, training/inference/pretraining cost separately.',
        updates=0, minimum_gpus=1,
        prerequisites=['Numerical checks and exact workload/backend/dtype/shape must be fixed first.',
                       'Measure rather than infer performance from FLOPs; do not perform formal training.']),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def source_integrity():
    pins = read(ROOT / 'evidence/sources.json')
    count = 0
    for name, repo in pins.items():
        for file, expected in repo.get('selected_files', {}).items():
            if sha(ROOT / 'sources' / name / file) != expected:
                raise ValueError('Author source changed: ' + name + '/' + file)
            count += 1
    return dict(repositories=len(pins), selected_files=count,
                manifest_sha256=sha(ROOT / 'evidence/sources.json'))


def baseline_evidence():
    summary = read(BASELINE)
    if summary['status'] != 'passed_scoped_checks':
        raise ValueError('CPU baseline is not passed_scoped_checks')
    if summary['configuration_manifest_sha256'] != sha(ROOT / 'configs/paired_experiments/manifest.json'):
        raise ValueError('CPU baseline does not bind the current paired configuration manifest')
    for file, expected in summary['code_sha256'].items():
        if sha(ROOT / file) != expected:
            raise ValueError('CPU baseline implementation changed: ' + file)
    rows = [row for values in summary['training'].values() for row in values] + summary['other_checks']
    evidence = []
    for row in rows:
        path = ROOT / row['evidence']
        report = read(path)
        if row['status'] not in ('byte_exact', 'passed') or report['status'] != row['status']:
            raise ValueError('CPU evidence status mismatch: ' + str(path))
        if report.get('gpu_used') is not False:
            raise ValueError('Baseline is not CPU-only: ' + str(path))
        evidence.append(dict(profile=row['profile'], path=row['evidence'],
                             status=row['status'], sha256=sha(path)))
    return dict(summary=str(BASELINE.relative_to(ROOT)), sha256=sha(BASELINE),
                implementation_files_verified=len(summary['code_sha256']),
                reports=evidence, scope='Prior scoped CPU evidence; no new model run in this preparation audit.')


def build_plan(bindings):
    manifest = load_manifest()
    if set(manifest['papers']) != set(SPECIAL):
        raise ValueError('Update the per-paper GPU specifications for the changed paper list')
    profiles = []
    for entry in manifest['profiles']:
        modes = ['paper'] + (['fair-rdd'] if entry['fair_eligible'] else [])
        plans = {mode: compile_plan(manifest, entry, mode, bindings) for mode in modes}
        # Reference argv must not change with python/python3 symlinks or when
        # a different CPU interpreter is used to audit this static bundle.
        # The future certified runner binds its own checked environment.
        for plan in plans.values():
            for backend in ('original', 'unified'):
                plan['production_entrypoints'][backend]['argv'][0] = '@experiment_python@'
        profiles.append(dict(id=entry['id'], paper=entry['paper'], config=entry['config'],
            config_sha256=entry['config_sha256'], native_precision=entry['paper_contract']['optimizer'].get('precision', 'float32'),
            contracts={mode: dict(sha256=p['contract_sha256'], controls=p['contract']['controls'],
                required_assets=p['required_assets'], missing_assets=p['missing_assets'],
                blocked_reasons=p['blocked_reasons'], data_binding=p['contract']['data'],
                author_cli_reference=p['production_entrypoints']) for mode, p in plans.items()}))
    methods = []
    for name, metadata in manifest['papers'].items():
        special, topology, assets = SPECIAL[name]
        categories = ['cuda_sample_decode', 'cuda_cost'] if name in INFERENCE_ONLY else list(CATEGORIES)
        selected_profiles = [row['id'] for row in profiles if row['paper'] == name]
        methods.append(dict(id=name, title=metadata['title'], source=metadata.get('repository'),
            profiles=selected_profiles, special_checks=special, native_topology=topology,
            local_assets=assets, paper_recipe_gaps=metadata['gaps'], cases=[dict(
                id=name + '/' + category, category=category, **CATEGORIES[category],
                status='specification_prepared_runner_pending', gpu_executed=False,
                blocked_reasons=['Current user instruction: CPU only.',
                                 'This bundle does not contain a certified CUDA/native-CLI execution adapter.'])
                for category in categories]))
    return dict(version=1, device_policy=dict(authorized='cpu', gpu_execution_enabled=False,
                downloads_enabled=False, formal_training_enabled=False,
                quote='只用cpu。但是如果有一定要用gpu的地方那就一次性全部准备好'),
        scope='14 executable external releases plus local RDD; 33 paired training templates and three inference-only releases.',
        status='specifications_prepared_not_gpu_certified', paired_manifest_sha256=sha(ROOT / 'configs/paired_experiments/manifest.json'),
        source_integrity=source_integrity(), cpu_baseline=baseline_evidence(), profiles=profiles, methods=methods,
        unreleased=manifest['unreleased'],
        acceptance=dict(same_cuda_execution_path='Require byte equality first; report every mismatch, finite/NaN/Inf counts, first layer/stage, max absolute/relative and RMS error.',
            different_kernel='Do not auto-pass with arbitrary global tolerances. Isolate a source backend switch and per-operation rounding error; masks/layout/math must be identical. Record numerical equivalence separately from byte equality.',
            stochastic_coupling='For algorithm parity, supply the SAME noise/times/posterior draws or restore all RNG states on the SAME device. CPU vs CUDA RNG streams need not generate the same samples from one seed.',
            distributed='Different all-reduce orders may round differently; report rank-local and reduced gradients separately. Do not claim equality with the historical 8-rank RDD from the 1-process fair recipe.',
            quality='No short GPU test certifies FID or paper-best retraining quality. Full real-data training and matching evaluation are separate, currently unstarted tasks.'),
        future_execution=dict(authorization='Not present; prepare only.', host='surfi1',
            physical_gpu_selection=None, absolute_deadline=None,
            independent_guard='multiscale_diffusion/experiments/official_video_baselines/deadline_guard.py',
            independent_guard_sha256=sha(ROOT.parent/'experiments/official_video_baselines/deadline_guard.py'),
            resource_note='No old card/deadline/80GiB requirement is renewed. Bind future explicitly authorized resources and estimate actual paired-state disk/RAM/VRAM first.',
            order=['Finish missing CPU adapters/config/asset checks.', 'Single-GPU numerical and decode checks, isolated source namespaces per process.',
                   'Multi-GPU native lifecycle only if separately supported and authorized.', 'Measure cost after correctness.'],
            retention='Hash/store JSON/log/config/trace first; remove temporary checkpoint/media payloads after validation. Keep user data, original weights and formal results.'))


def audit(plan):
    assert plan['device_policy']['authorized'] == 'cpu'
    assert plan['device_policy']['gpu_execution_enabled'] is False
    assert len(plan['profiles']) == 33 and len(plan['methods']) == 15
    assert len({row['id'] for row in plan['profiles']}) == 33
    assert sum('fair-rdd' in row['contracts'] for row in plan['profiles']) == 11
    case_ids = [case['id'] for method in plan['methods'] for case in method['cases']]
    assert len(case_ids) == len(set(case_ids)) == 66
    assert all(not case['gpu_executed'] for method in plan['methods'] for case in method['cases'])
    assert 'torch' not in sys.modules
    assert not os.environ['CUDA_VISIBLE_DEVICES']
    assert len(plan['cpu_baseline']['reports']) == 50
    return dict(status='passed_plan_audit', device='cpu', gpu_used=False,
        model_test_executed=False, formal_training_started=False, downloads_performed=False,
        methods=len(plan['methods']), paired_profiles=len(plan['profiles']), fair_profiles=11,
        gpu_specifications=len(case_ids), certified_cuda_runners=0,
        existing_cpu_reports_hashed=len(plan['cpu_baseline']['reports']),
        existing_implementation_files_verified=plan['cpu_baseline']['implementation_files_verified'],
        source_integrity=plan['source_integrity'],
        scope='Coverage, source/config/report hashes and execution policy; NOT new activation/gradient or CUDA evidence.')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action', choices=['plan', 'check'], default='plan')
    p.add_argument('--bindings', type=Path, help='Optional explicit local data/assets map, same schema as paired launchers.')
    a = p.parse_args()
    bindings = read(a.bindings) if a.bindings else {}
    plan = build_plan(bindings)
    report = audit(plan)
    if a.action == 'plan':
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / 'plan.json').write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n')
        report['plan_sha256'] = sha(OUT / 'plan.json')
        report['planner_sha256'] = sha(Path(__file__))
        report['wrapper_sha256'] = sha(ROOT / 'prepare_gpu_checks.sh')
        EVIDENCE.mkdir(parents=True, exist_ok=True)
        (EVIDENCE / 'SUMMARY.json').write_text(json.dumps(report, indent=2) + '\n')
    else:
        if read(OUT / 'plan.json') != plan:
            raise ValueError('Prepared plan is stale or bindings differ; run --action plan with the intended bindings.')
        saved = read(EVIDENCE / 'SUMMARY.json')
        if saved['plan_sha256'] != sha(OUT / 'plan.json') or saved['planner_sha256'] != sha(Path(__file__)) or saved['wrapper_sha256'] != sha(ROOT / 'prepare_gpu_checks.sh'):
            raise ValueError('Plan/planner/wrapper hash changed; regenerate and review the plan.')
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
