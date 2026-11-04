"""CPU scope audit: old DDPM parity is not FM fresh/noise-filling parity.

No production changes, checkpoints, data, GPU, or image-quality claims.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import torch

from umdiff.grids import GridMap
from umdiff.multires_process import RDDProcess
from umdiff.randomness import RandomStream

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'evidence/rdd_sampling_scope_20261104'
SOURCE = ROOT.parent/'diffusion/flow_matching_with_drift.py'
spec = importlib.util.spec_from_file_location('scope_old_fm', SOURCE)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
torch.set_num_threads(2)


def difference(a, b):
    return dict(equal=torch.equal(a, b), max_abs=(a-b).abs().max().item())


def boundary_case(seed, t):
    torch.manual_seed(seed)
    state = torch.randn(2, 4, 4, 4, dtype=torch.float64)
    clean_estimate = torch.randn_like(state)
    op = GridMap((8, 8), (4, 4))
    torch.manual_seed(seed+100)
    native = old.noise_filling_transition_fm(state, clean_estimate, t, 8)
    torch.manual_seed(seed+100)
    innovation = torch.randn_like(op.up(clean_estimate))
    recovered_noise = (state-t*clean_estimate)/(1-t)
    coupled = t*op.up(clean_estimate)+(1-t)*op.lift_noise(recovered_noise, innovation)
    fresh = t*op.up(clean_estimate)+(1-t)*innovation
    same_rule = difference(native, coupled)
    changed_rule = difference(native, fresh)
    # GridMap sums (low + high) - projection, old code low + (high - projection).
    # Algebra agrees; floating-point addition need not be byte-identical.
    torch.testing.assert_close(native, coupled, rtol=0, atol=2e-14)
    assert not changed_rule['equal']
    return dict(seed=seed, boundary=t, native_fm_vs_unified_coupled=same_rule,
                native_fm_vs_fresh=changed_rule)


class RandomField(torch.nn.Module):
    """Small nonzero spatial field; deliberately not a trained DiT."""
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(4, 4, 3, padding=1).double()
        with torch.no_grad():
            for p in self.parameters():
                p.normal_(0, .025)

    def forward(self, x, time, stage, **kwargs):
        return torch.tanh(self.conv(x)) + .01*time[:, None, None, None] + .005*stage


@torch.no_grad()
def trajectory_case(solver):
    torch.manual_seed(20261104)
    model = RandomField()
    initial = torch.randn(2, 4, 4, 4, dtype=torch.float64)
    traces = {}
    for boundary in ('fresh', 'coupled'):
        process = RDDProcess(grids=[[4, 4], [8, 8], [16, 16]],
                             steps=[4, 4, 4], bounds=[0., .3, .6, 1.],
                             solver=solver, boundary=boundary)
        final, trace = process.sample(model, initial.clone(), RandomStream(972), {}, True)
        traces[boundary] = (final, trace)
    differences = [dict(index=i, shape=list(a.shape), **difference(a, b))
                   for i, (a, b) in enumerate(zip(traces['fresh'][1], traces['coupled'][1]))]
    assert len(traces['fresh'][1]) == len(traces['coupled'][1]) == 15
    first = next(r['index'] for r in differences if not r['equal'])
    assert first == 5  # initial + four coarse updates; next is the first boundary.
    assert not torch.equal(traces['fresh'][0], traces['coupled'][0])
    return dict(solver=solver, first_different_snapshot=first,
                first_difference_is='first resolution boundary, after four identical coarse updates',
                snapshots=differences, final=difference(traces['fresh'][0], traces['coupled'][0]))


def report_summary(path):
    result = json.loads(path.read_text())
    checks = result['checks']
    assert result['status'] == 'passed' and all(c['byte_exact'] for c in checks)
    return dict(path=str(path.relative_to(ROOT)), checks=len(checks),
                max_abs=max(c['max_abs'] for c in checks),
                model_calls=result.get('model_calls'), details=result.get('details'))


if __name__ == '__main__':
    original = sorted((ROOT/'evidence/sampling_parity_20261015/final_common').glob(
        'seed*_batch*/*rdd_ddpm_xl_legacy_euler.json'))
    assert len(original) == 4
    boundaries = [boundary_case(s, t) for s, t in ((641, .3), (972, .6), (300, .4))]
    trajectories = [trajectory_case(s) for s in ('euler', 'heun')]
    source_files = [SOURCE, ROOT/'umdiff/multires_process.py', ROOT/'umdiff/grids.py',
                    ROOT/'tools/validate_comparison_sampling.py', Path(__file__)]
    report = dict(status='passed', device='cpu', dtype='float64 for new FM probes',
                  historical_ddpm=[report_summary(p) for p in original],
                  current_ddpm_recheck=report_summary(OUT/'ddpm_recheck/main_rdd_ddpm_xl_legacy_euler.json'),
                  fm_boundary_checks=boundaries, fm_negative_controls=trajectories,
                  scope='Historical reports certify DDPM fresh-vs-fresh. New probes check the old FM boundary helper versus unified coupled algebra and demonstrate fresh/coupled trajectory differences. They do not certify a complete original-vs-unified FM sampling pipeline or trained image quality.',
                  coupled_roundoff='Old helper adds low + (high - projection); GridMap adds (low + high) - projection. This can differ at FP64 roundoff; it is not byte parity.',
                  source_sha256={str(p.relative_to(ROOT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'AUDIT.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(status=report['status'], boundaries=boundaries,
                          trajectories=[{k:v for k,v in r.items() if k != 'snapshots'} for r in trajectories])))
