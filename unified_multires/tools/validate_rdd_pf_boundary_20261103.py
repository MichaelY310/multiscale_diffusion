"""CPU algebra/endpoint checks for an RDD path with a PF-style boundary.

No production registration, checkpoint adaptation, learned model, or quality test.
The reference is the existing controlled orthogonal PF transition, not an author
pipeline or a new native parity certificate.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from umdiff.multires_process import PyramidFlowProcess
from umdiff.paths import RDDFlow


class FixedNoise:
    def __init__(self, value):
        self.value = value

    def normal(self, reference):
        assert reference.shape == self.value.shape
        return self.value.to(reference)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve prior evidence; select a new output')
    torch.set_num_threads(1)
    torch.manual_seed(31104)
    errors, algebra, components = [], [], []
    for volume in (2, 4, 8):
        up = np.ones((volume, 1))
        projector = up@up.T/volume
        complement = np.eye(volume)-projector
        omega = math.sqrt(volume)
        for old_signal in (.2, .4, .7, .9):
            for schedule in ('fm', 'vp'):
                if schedule == 'fm':
                    old_sigma = 1-old_signal
                    new_signal = old_signal/(omega-(omega-1)*old_signal)
                    new_sigma = 1-new_signal
                else:
                    old_sigma = math.sqrt(1-old_signal**2)
                    new_signal = old_signal/math.sqrt(volume-(volume-1)*old_signal**2)
                    new_sigma = math.sqrt(1-new_signal**2)
                scale = new_signal/old_signal
                old_covariance = scale**2*old_sigma**2*(up@up.T)
                added_covariance = new_sigma**2*complement
                total = old_covariance+added_covariance
                required = new_sigma**2*np.eye(volume)
                residual = float(np.max(np.abs(total-required)))
                errors.append(residual)
                assert np.linalg.eigvalsh(added_covariance).min() > -1e-12
                unchanged_covariance = old_sigma**2*(np.eye(volume)-up@up.T)
                unchanged_min = float(np.linalg.eigvalsh(unchanged_covariance).min())
                errors.append(abs(unchanged_min-(1-volume)*old_sigma**2))
                assert unchanged_min < 0
                # Characteristic functions: a non-Gaussian (discrete) clean law.
                # This is an example supporting the conditional-Gaussian proof,
                # not an empirical inference of equality for arbitrary priors.
                prior_values = np.array([-1.3, .2, 2.1])
                prior_weights = np.array([.25, .4, .35])
                max_cf_error = 0.
                for frequency in (np.arange(volume)/volume, np.linspace(-1, 1, volume)):
                    phase = frequency@(new_signal*up)
                    clean_cf = np.sum(prior_weights*np.exp(1j*phase[0]*prior_values))
                    actual_cf = clean_cf*np.exp(-.5*frequency@total@frequency)
                    target_cf = clean_cf*np.exp(-.5*frequency@required@frequency)
                    max_cf_error = max(max_cf_error, float(abs(actual_cf-target_cf)))
                errors.append(max_cf_error)
                algebra.append({'volume':volume, 'schedule':schedule,
                                'old_signal':old_signal, 'new_signal':new_signal,
                                'new_sigma':new_sigma, 'noise_covariance_error':residual,
                                'non_gaussian_clean_cf_error':max_cf_error,
                                'unchanged_entry_required_covariance_min_eigenvalue':unchanged_min})
    for grid in ((2, 3), (3, 2, 3)):
        fine_grid = (*grid[:-2], grid[-2]*2, grid[-1]*2)
        reference = PyramidFlowProcess([grid, fine_grid], network_clock='local', boundary_policy='orthogonal')
        op = reference.maps[0]
        fine_clean = torch.randn(2, 2, *fine_grid, dtype=torch.float64)
        coarse_clean = op.down(fine_clean)
        coarse_noise = torch.randn_like(coarse_clean)
        innovation = torch.randn_like(fine_clean)
        for old_signal in (.2, .4, .7, .9):
            old_sigma = 1-old_signal
            new_signal = old_signal/(2-old_signal)
            new_sigma = 1-new_signal
            old_state = old_signal*coarse_clean+old_sigma*coarse_noise
            # Inject only the chosen endpoint into the reference fixture. This
            # isolates its kernel; it does not test the native discretized clock.
            reference.ends[0] = old_sigma
            actual = reference.transition(old_state, 0, FixedNoise(innovation))
            proposed = op.up(old_state)/(2-old_signal)+new_sigma*(innovation-op.project(innovation))
            reconstructed_noise = op.up(coarse_noise)/2+innovation-op.project(innovation)
            expected = new_signal*op.up(coarse_clean)+new_sigma*reconstructed_noise
            path = RDDFlow(start=new_signal, end=1.)
            path.projection = op
            entry = path.sample(fine_clean, torch.full((2,),new_signal,dtype=torch.float64), reconstructed_noise)
            middle_t = (new_signal+1)/2
            middle = path.sample(fine_clean, torch.full((2,),middle_t,dtype=torch.float64), innovation)
            tau = (middle_t-new_signal)/(1-new_signal)
            coarse = op.project(fine_clean)
            rdd_expected = middle_t*(coarse+tau*(fine_clean-coarse))+(1-middle_t)*innovation
            pf_middle = middle_t*coarse+tau*(fine_clean-coarse)+(1-middle_t)*innovation
            endpoint = path.sample(fine_clean, torch.ones(2,dtype=torch.float64), innovation)
            residuals = {
                'existing_pf_transition_vs_proposed':float((actual-proposed).abs().max()),
                'proposed_vs_training_entry_coupling':float((proposed-expected).abs().max()),
                'actual_rdd_path_entry_vs_transition':float((entry.state-proposed).abs().max()),
                'actual_rdd_interior_vs_rdd_formula':float((middle.state-rdd_expected).abs().max()),
                'rdd_final_endpoint':float((endpoint.state-fine_clean).abs().max()),
            }
            difference = float((middle.state-pf_middle).abs().max())
            assert difference > 1e-5, 'The RDD interior must remain distinct from PF'
            errors.extend(residuals.values())
            components.append({'coarse_grid':grid, 'fine_grid':fine_grid,
                               'old_signal':old_signal, 'new_signal':new_signal,
                               'residuals':residuals, 'rdd_vs_pf_interior_max_difference':difference})
    assert max(errors) < 2e-12, max(errors)
    assert not torch.cuda.is_initialized()
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'unified_multires/umdiff/paths.py',
             root/'unified_multires/umdiff/multires_process.py', root/'unified_multires/umdiff/grids.py']
    result = {'date_la':'2026-11-03', 'date_utc':'2026-11-04',
              'scope':'CPU float64 covariance/characteristic-function/endpoint checks; existing controlled PF transition and RDDFlow. No learned forward/backward, native author pipeline, generation-quality result, or ready-to-run hybrid training config.',
              'assumptions':'Nearest repeat of m children per coarse value; block-average projection; independent Gaussian innovation; correct incoming marginal; matched next-stage signal/noise coefficients.',
              'algebra_cases':algebra, 'component_cases':components,
              'max_identity_residual':max(errors), 'cuda_initialized':False,
              'source_sha256':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'algebra_cases':len(algebra),'component_cases':len(components),
                      'max_identity_residual':max(errors),'cuda_initialized':False}))


if __name__ == '__main__':
    main()
