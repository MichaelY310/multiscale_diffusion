"""CPU algebra audit, not trained-model parity or a quality experiment.

Compare existing path components at deliberately aligned stage endpoints.
The PF endpoint expression is transcribed from paper Eqs. 9--11 and the
locally pinned author source; this does not execute its complete pipeline.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from umdiff.paths import RDDFlow, EndpointFlow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    generator = torch.Generator(device='cpu').manual_seed(20261102)
    rows = []
    for shape in ((1, 2, 8, 8), (1, 2, 3, 8, 8)):
        for a, b in ((.4, .7), (.7, 1.)):
            rdd = RDDFlow(start=a, end=b)
            pf = EndpointFlow(start=a, end=b, velocity_clock='global')
            x = torch.randn(shape, generator=generator, dtype=torch.float64)
            eps = torch.randn(shape, generator=generator, dtype=torch.float64)
            coarse = rdd.projection(x)
            detail = x-coarse
            for tau in (0., .25, .5, .75, 1.):
                t = torch.tensor([a+tau*(b-a)], dtype=torch.float64)
                r = rdd.sample(x, t, eps)
                p = pf.sample(x, t, eps)
                L = b-a
                native_start = a*coarse+(1-a)*eps
                native_end = b*x+(1-b)*eps
                native_expr = (1-tau)*native_start+tau*native_end
                # Author target is d/d(rho), rho=1-tau; divide by -L.
                native_global_velocity = -(native_start-native_end)/L
                expected_gap = L*tau*(1-tau)*detail
                expected_velocity_gap = (2*tau-1)*detail
                # Both paths admit a band-wise inverse, with different factors.
                k_pf = b*(1-a)/L
                y_pf = p.state+(1-t)*p.velocity
                py = pf.projection(y_pf)
                pf_inverse = py+(y_pf-py)/k_pf
                checks = {
                    'pf_endpoint_expression': (p.state-native_expr).abs().max().item(),
                    'pf_global_velocity': (p.velocity-native_global_velocity).abs().max().item(),
                    'state_gap_identity': (p.state-r.state-expected_gap).abs().max().item(),
                    'velocity_gap_identity': (r.velocity-p.velocity-expected_velocity_gap).abs().max().item(),
                    'rdd_clean_inverse': (rdd.clean_from_velocity(r.state,r.velocity,t)-x).abs().max().item(),
                    'pf_component_clean_inverse': (pf.clean_from_velocity(p.state,p.velocity,t)-x).abs().max().item(),
                    'pf_explicit_clean_inverse': (pf_inverse-x).abs().max().item(),
                }
                assert max(checks.values()) < 2e-12, checks
                rows.append({'shape':list(shape), 'stage':[a,b], 'tau':tau,
                             'state_difference':(p.state-r.state).abs().max().item(),
                             'velocity_difference':(r.velocity-p.velocity).abs().max().item(),
                             'max_abs_residuals':checks})
    # Exact 4-pixel noise covariance algebra; no Monte Carlo needed.
    eye = torch.eye(4, dtype=torch.float64)
    uuT = torch.ones((4,4), dtype=torch.float64)
    e = .7
    s = e/(2-e)
    k = s/e
    correction = (1-s)**2*eye-k*k*(1-e)**2*uuT
    covariance_error = (k*k*(1-e)**2*uuT+correction-(1-s)**2*eye).abs().max().item()
    assert covariance_error < 2e-12
    assert torch.linalg.eigvalsh(correction).min() > -2e-12
    same_clock = (1-e)**2*(eye-uuT)
    assert torch.linalg.eigvalsh(same_clock).min() < 0
    root = Path(__file__).resolve().parents[2]
    pfroot = root.parent/'official_video_baselines_20261004/reference/Pyramid-Flow'
    sources = [Path(__file__), root/'unified_multires/umdiff/paths.py',
               root/'unified_multires/umdiff/operators.py',
               root/'unified_multires/umdiff/multires_process.py',
               root/'multiscale_diffusion/diffusion/flow_matching_with_drift.py',
               root/'multiscale_diffusion/diffusion/gaussian_diffusion_with_drift.py',
               pfroot/'pyramid_dit/pyramid_dit_for_video_gen_pipeline.py',
               pfroot/'diffusion_schedulers/scheduling_flow_matching.py']
    result = {'scope':'CPU float64 algebra; no model forward, training, native full pipeline or quality test',
              'torch_version':torch.__version__, 'cuda_initialized':torch.cuda.is_initialized(),
              'aligned_stage_cases':rows,
              'boundary':{'coarse_end':e,'fine_start':s,'covariance_error':covariance_error,
                          'correction_eigenvalues':torch.linalg.eigvalsh(correction).tolist(),
                          'same_clock_correction_eigenvalues':torch.linalg.eigvalsh(same_clock).tolist()},
              'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'max_algebra_residual':max(max(r['max_abs_residuals'].values()) for r in rows),
                      'cuda_initialized':result['cuda_initialized'],'boundary':result['boundary']}))


if __name__ == '__main__':
    main()
