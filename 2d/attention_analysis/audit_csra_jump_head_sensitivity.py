#!/usr/bin/env python3
"""CPU-only audit of the ACTUAL preview sampler, without changing old behavior.

Uses synthetic output interventions through the real jump sampler. The measured
gains map same-input main-head error to one next-state error, NOT terminal quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from sample_rdd_stagewide_block_reuse_preview import make_diffusions


def audit():
    torch.set_num_threads(2)
    torch.manual_seed(173)
    diffusion = make_diffusions([80, 80, 90])[0]
    x = torch.randn(1, 4, 32, 32)
    zero = torch.zeros_like(x)
    low = torch.ones_like(x)
    high = low.clone()
    high[..., 1::2, :] = -1
    # Unit RMS cell-constant / within-cell-difference fields, respectively.
    rows = []
    for i in range(diffusion.num_timesteps - 1, -1, -1):
        t, nxt = torch.tensor([i]), torch.tensor([max(0, i-1)])
        def result(main, var, stochastic=False):
            torch.manual_seed(190)
            return diffusion.jump_step_sample(
                lambda *a, **kw: torch.cat((main, var), 1), x, t, nxt,
                clip_denoised=False, deterministic=not stochastic)
        base = result(zero, zero)
        # _sample_batch_unified returns samples, NOT pred_xstart. Consequently
        # the final local-0 fine call computes a prediction but leaves sample
        # unchanged. Intermediate-stage pred_xstart DOES feed upsampling.
        # Audit existing behavior; do not silently fix or reinterpret it.
        key = 'sample'
        gains = []
        max_deviation = 0.
        for pattern in (low, high):
            delta = result(pattern, zero)[key] - base[key]
            gain = float((delta * pattern).mean())
            max_deviation = max(max_deviation, float((delta-gain*pattern).abs().max()))
            gains.append(gain)
        torch.testing.assert_close(result(zero, -low, True)['sample'],
                                   result(zero, low, True)['sample'], atol=0, rtol=0)
        torch.testing.assert_close(result(zero, -low, True)['pred_xstart'],
                                   result(zero, low, True)['pred_xstart'], atol=0, rtol=0)
        # Check a general field too, not only the two special basis vectors.
        error = torch.randn_like(x)
        pool = torch.nn.functional.avg_pool2d(error, 2).repeat_interleave(2,-2).repeat_interleave(2,-1)
        expected = gains[0]*pool + gains[1]*(error-pool)
        actual = result(error, zero)[key]-base[key]
        torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-4)
        rows.append(dict(local_t=i, t=int(diffusion.timestep_map[i]),
                         t_next=int(diffusion.timestep_map[max(0,i-1)]),
                         returned_quantity=key, low_gain=gains[0], high_gain=gains[1],
                         basis_max_abs_error=max_deviation,
                         mixed_field_max_abs_error=float((actual-expected).abs().max())))
    t = torch.tensor([45])
    def pm(var):
        return diffusion.p_mean_variance(lambda *a, **kw: torch.cat((zero,var),1),
                                          x,t,clip_denoised=False)
    logvar_change = float((pm(low)['log_variance']-pm(-low)['log_variance']).abs().max())
    if logvar_change <= 0: raise AssertionError('p_mean_variance must still use learned variance')
    return dict(model_var_type=str(diffusion.model_var_type), rows=rows,
                p_mean_variance_logvar_change=logvar_change,
                jump_sample_variance_intervention_max_abs=0.,
                all_90_times_variance_parity=True,
                warning='LEARNED_RANGE metadata and p_mean_variance use the variance head, but jump_step_sample ignores that returned variance and uses analytic Sigma. This does NOT describe ordinary p_sample. Gains are local same-input perturbations, not final-FID sensitivity or global Jacobians. Spatial bands are 2x2 average/difference, not ideal Fourier bands. The caller returns sample, so final fine local0 is an identity update despite a model call. Intermediate coarse/middle pred_xstart still feeds boundary upsampling. No sampler or checkpoint modified; CPU synthetic tensors only.')


def main(args):
    destination = Path(args.output)
    if destination.exists(): raise FileExistsError(destination)
    data = audit()
    by_time = {r['t']:r for r in data['rows']}
    transformed = []
    root = Path('eval_results/csra_prefix8_conditional_rollout_shift_20260913')
    for path in sorted(root.glob('seed_*.json')):
        old = json.loads(path.read_text())
        for r in old['rows']:
            gain = by_time[r['t']]
            transformed.append(dict(seed=old['seed'], arm=old['arm'], **r,
                local_output_low_mse=r['low_mse']*gain['low_gain']**2,
                local_output_high_mse=r['high_mse']*gain['high_gain']**2,
                returned_quantity=gain['returned_quantity']))
    if len(transformed) != 256: raise ValueError('Expected four completed shift files, 64 rows each')
    data['transformed_development_rows'] = transformed
    files = ['diffusion/gaussian_diffusion_with_drift.py', '2d/sample_fid_unified.py',
             '2d/attention_analysis/sample_rdd_stagewide_block_reuse_preview.py']
    data['source_sha256'] = {p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in files}
    data['complete'] = True
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, indent=2)+'\n')
    print(json.dumps({k:v for k,v in data.items() if k not in ('rows','transformed_development_rows')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='eval_results/csra_prefix8_jump_sensitivity_20260913/metrics.json')
    main(p.parse_args())
