"""CPU differential check of version3's RDD-style outer objective.

The reference enumerates pre-existing single-phase objectives independently.
It is an aggregation/optimizer test, not a claim of original-paper equivalence.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import random
import traceback
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from umdiff import build
from umdiff.cli import make_method, setup
from umdiff.engine import TrainingResult
from umdiff.randomness import RandomStream
from umdiff.time_sampling import DFMTimes, LapFlowTimes
from activation_trace import ActivationAudit, byte_view

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / 'configs/comparison_imagenet256'


def helpers():
    # This historical script has a top-level runner: load functions only.
    path = ROOT / 'tools/validate_main_comparison.py'
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
    namespace = {'__file__': str(path), 'ROOT': ROOT, 'CONFIGS': CONFIGS}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def equal(a, b, path='value'):
    if torch.is_tensor(a):
        assert a.shape == b.shape and a.dtype == b.dtype, path
        assert torch.equal(byte_view(a), byte_view(b)), path
    elif isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), path
    elif isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a: equal(a[k], b[k], f'{path}/{k}')
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b), path
        for i, v in enumerate(a): equal(v, b[i], f'{path}/{i}')
    else:
        assert a == b, path


def rng():
    return torch.get_rng_state(), np.random.get_state(), random.getstate()


def restore(state):
    torch.set_rng_state(state[0]); np.random.set_state(state[1]); random.setstate(state[2])


class ManualAggregate:
    """No call to the new aggregator or its reduction helper on the reference."""
    def __init__(self, config):
        self.config = config
        single = copy.deepcopy(config)
        single.pop('training_aggregation', None)
        self.method = make_method(single)

    def __getattr__(self, name):
        return getattr(self.method, name)

    def training(self, model, images, t, stream, noise=None, model_kwargs=None, active=None, context=None):
        kwargs = model_kwargs or {}; context = context or {}; results = []
        if 'training_aggregation' in self.config:
            name = self.config['batch']['name']
            for i, spec in enumerate(self.config['training_aggregation']['samplers']):
                args = copy.deepcopy(spec['args']); args.pop('stage', None)
                if name == 'dfm':
                    with patch('numpy.random.choice', return_value=i):
                        clock, extra = DFMTimes(**args)(len(images), images.device, stream)
                elif name == 'lapflow':
                    with patch('random.randint', return_value=i):
                        clock, extra = LapFlowTimes(**args)(len(images), images.device, stream)
                else:
                    table = self.config['batch']['args']['resolutions']
                    size = sorted(set(table))[i]
                    indices = [j for j, r in enumerate(table) if r == size]
                    index = stream.randint(0, len(indices), (1,), device=images.device)
                    clock = torch.tensor(indices, device=images.device)[index].expand(len(images))
                    extra = {}
                results.append(self.method.training(model, images, clock, stream,
                               model_kwargs=kwargs, context={**context, **extra}))
        elif self.config.get('task') == 'multires':
            p = self.method.process
            if self.config['process']['name'] == 'nonuniform_haar':
                # Existing phase kernel supplies per-phase DSM vectors; ignore
                # its old vector total and independently reduce those vectors.
                p.stage_reduction = 'sum_vectors'
                result = self.method.training(model, images, t, stream, noise, kwargs, active, context)
                phase_losses = [result.terms[f'stage{i}/weighted_dsm'] for i in range(len(p.grids))]
                return TrainingResult(sum(v.mean() for v in phase_losses), result.prediction,
                                      result.target, result.state, {})
            clean = p.clean_pyramid(images, context)
            eps = p.noise_pyramid(stream.normal(images) if noise is None else noise)
            for i in range(len(clean)):
                u = stream.uniform((len(images),), device=images.device).to(images.dtype)
                state, target, clock, meta = p.prepare(clean, eps[i], u, i)
                prediction = p.call(model, state, clock, i, kwargs)
                loss, terms = p.loss(prediction, target, meta)
                results.append(TrainingResult(loss, prediction, target, state, terms))
        else:
            return self.method.training(model, images, t, stream, noise, kwargs, active, context)
        return TrainingResult(sum(r.loss.mean() for r in results), [r.prediction for r in results],
                              [r.target for r in results], [r.state for r in results], {})


def phase_count(config):
    return len(config.get('training_aggregation', {}).get('samplers', [])) or len(config.get('grids', [])) or len(config.get('training_stages', [])) or 1


def test_recipe(path, out, helper):
    spec = json.loads(path.read_text()); c = helper['small'](spec)
    case_id = spec['suite']+'_'+spec['id']
    # Two microbatches of two DIFFERENT images catch mean/sum order bugs.
    c['training']['optimizer']['accumulation_steps'] = 2
    method, net, trainer = setup(c, torch.device('cpu'), 617)
    with torch.no_grad():
        for parameter in net.parameters(): parameter.normal_(0, .03)
    trainer.ema.load_state_dict(net.state_dict())
    reference = copy.deepcopy(trainer)
    reference.method = ManualAggregate(c)
    clock = build('time_sampler', c['training']['time_sampler'])
    reference.time_sampler = copy.deepcopy(clock); trainer.time_sampler = clock
    images = torch.randn(4, 4, 32, 32); labels = torch.tensor([3, 11, 19, 47])
    audit = ActivationAudit(case_id, [('network', reference.model, net)], out)
    losses = []
    update_calls = [0]
    handle = trainer.optimizer.register_step_post_hook(lambda *args: update_calls.__setitem__(0, update_calls[0]+1))
    for step in range(2):
        state = rng(); audit.begin(step); audit.side('native')
        expected = reference.step(images, None, model_kwargs={'y': labels}, time_sampler=reference.time_sampler)
        expected_rng = rng(); restore(state); audit.side('unified')
        got = trainer.step(images, None, model_kwargs={'y': labels}, time_sampler=clock)
        audit.parameters(reference.model, net); audit.finish_segment()
        for key in ('loss', 'prediction', 'target', 'state'):
            equal(getattr(expected, key), getattr(got, key), key)
        for key in ('model', 'ema', 'optimizer', 'stream', 'step', 'images_seen'):
            equal(reference.state_dict()[key], trainer.state_dict()[key], key)
        equal(expected_rng, rng(), 'global_rng')
        assert trainer.step_index == step+1 and trainer.images_seen == 4*(step+1)
        if phase_count(c) > 1 and 'training_stages' not in c:
            for micro in got.microbatches:
                objectives = [v for k, v in micro.terms.items() if k.endswith('/objective')]
                assert len(objectives) == phase_count(c)
                equal(micro.loss, sum(objectives), 'diagnostic_stage_sum')
        losses.append(got.loss.mean().item())
    handle.remove(); assert update_calls == [2]
    report = audit.close()
    (out / case_id / 'LAYERS.json').write_text(json.dumps(report, indent=2)+'\n')
    assert not report['structural_mismatches'] and not report['first_differences']
    assert all(v['pairs'] == v['byte_exact'] for v in report['totals'].values())

    # Save, load into a newly constructed trainer, then compare the NEXT update.
    checkpoint = out / (case_id+'.pt'); trainer.save(checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest(); size = checkpoint.stat().st_size
    result = trainer.step(images, None, model_kwargs={'y': labels}, time_sampler=clock)
    expected = copy.deepcopy(trainer.state_dict())
    _, _, resumed = setup(c, torch.device('cpu'), 918)
    resumed.time_sampler = build('time_sampler', c['training']['time_sampler']); resumed.load(checkpoint)
    result2 = resumed.step(images, None, model_kwargs={'y': labels}, time_sampler=resumed.time_sampler)
    equal(result.loss, result2.loss, 'resume_loss'); equal(expected, resumed.state_dict(), 'resume_state')
    checkpoint.unlink()
    row = dict(id=spec['id'], suite=spec['suite'], status='passed', phases=phase_count(c), losses=losses,
               layers=report['totals'], optimizer_updates_compared=2, images_per_update=4,
               microbatches_per_update=2, resumed_next_update_exact=True,
               checkpoint=dict(sha256=digest, bytes=size, deleted_after_verification=True))
    del trainer, reference, resumed, report, audit; gc.collect()
    return row


def test_ssd_boundaries(helper):
    rows = []
    for path in CONFIGS.glob('*/ssd*.json'):
        spec = json.loads(path.read_text()); c = helper['small'](spec)
        method = make_method(c); table = c['batch']['args']['resolutions']
        indices = [i for s in c['training_aggregation']['samplers'] for i in s['args']['indices']]
        assert sorted(indices) == list(range(len(table)))
        transitions = [i for i in range(1, len(table)) if table[i] != table[i-1]]
        x = torch.randn(2, 4, 32, 32)
        for i in transitions:
            sampler = next(s for s in method.aggregation_samplers if i in s.indices)
            stream = RandomStream(17)
            with patch.object(stream, 'randint', return_value=torch.tensor([sampler.indices.index(i)])):
                clock, ctx = sampler(2, x.device, stream)
            assert clock.tolist() == [i, i]
            a = method.batch(x, clock, RandomStream(19), None, ctx)
            b = method.batch(x, torch.full((2,), i), RandomStream(19), None, {})
            equal(a.state, b.state); equal(a.target, b.target)
        rows.append(dict(id=spec['id'], all_training_indices_covered_once=True, transitions=transitions))
    return rows


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output; out.mkdir(parents=True, exist_ok=True)
    if (out/'TRAINING.json').exists(): raise ValueError('Use a new output directory to retain previous reports')
    torch.set_num_threads(2); helper = helpers()
    result = {'device':'cpu', 'scope':'Manual phase reduction vs production, common width32/depth2 FP32; full latent grids; no author outer-objective equality claim', 'configs':[]}
    for entry in json.loads((CONFIGS/'manifest.json').read_text())['entries']:
        try: row = test_recipe(CONFIGS/entry['file'], out, helper)
        except Exception: row = dict(id=entry['id'], status='failed', traceback=traceback.format_exc())
        result['configs'].append(row); print(json.dumps(row), flush=True)
        (out/'TRAINING.json').write_text(json.dumps(result, indent=2)+'\n')
    result['ssd_boundaries'] = test_ssd_boundaries(helper)
    result['legacy_rdd_parity'] = helper['rdd_parity']()
    result['cuda_initialized'] = torch.cuda.is_initialized()
    result['status'] = 'passed' if all(r['status']=='passed' for r in result['configs']) and not result['cuda_initialized'] else 'failed'
    (out/'TRAINING.json').write_text(json.dumps(result, indent=2)+'\n')
    return 0 if result['status']=='passed' else 1


if __name__ == '__main__': raise SystemExit(main())
