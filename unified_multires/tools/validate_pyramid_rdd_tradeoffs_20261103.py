"""CPU analytical examples only: not a learned model, parity, or quality test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.integrate import quad

from research.rdd_fm_integrator import Stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve earlier evidence; choose another output')
    residuals, rows, actions = [], [], []
    prior = np.array([[1., .6], [.6, 1.7]])
    clean = np.array([-.3, 1.2])
    noise = np.array([.4, -.6])
    for a, b in ((.15, .55), (.4, .7), (.7, 1.)):
        length = b-a
        stage = Stage(a, b)
        pf_derivative = np.array([1., b/length])
        pf_action = quad(lambda t: float(np.sum((pf_derivative*clean-noise)**2)), a, b)[0]
        rdd_action = quad(lambda t: float(np.sum((stage.coefficients(t)[1]*clean-noise)**2)), a, b)[0]
        action_gap = length*clean[1]**2/3
        residuals.append(abs((rdd_action-pf_action)-action_gap))
        actions.append({'stage': [a,b], 'pf_action': pf_action, 'rdd_action': rdd_action,
                        'gap_formula': action_gap})
        crossing = 1-np.sqrt((1-a)*(1-b))
        crossing_B = stage.coefficients(crossing)[3][1]
        residuals.append(abs(crossing_B-b*(1-a)/length))
        for fraction in (.1, .5, .9):
            t = a+fraction*length
            ar, dar, sigma, br = stage.coefficients(t)
            ap = np.array([t, b*fraction])
            bp = ap+sigma*pf_derivative
            ratio = t/b
            degradation = np.diag([1., ratio])
            extra_noise = np.diag([0., sigma**2*(1-ratio**2)])
            AR, AP = np.diag(ar), np.diag(ap)
            cov_r = AR@prior@AR.T+sigma**2*np.eye(2)
            cov_p = AP@prior@AP.T+sigma**2*np.eye(2)
            residuals.extend([
                float(np.max(np.abs(degradation@AP-AR))),
                float(np.max(np.abs(degradation@cov_p@degradation.T+extra_noise-cov_r))),
                float(np.max(np.abs(prior@AP.T@degradation.T-prior@AR.T))),
                abs((ar[1]/ap[1])**2-ratio**2),
            ])
            posterior_r = prior-prior@AR.T@np.linalg.solve(cov_r, AR@prior)
            posterior_p = prior-prior@AP.T@np.linalg.solve(cov_p, AP@prior)
            posterior_gap = posterior_r-posterior_p
            gap_eigenvalues = np.linalg.eigvalsh((posterior_gap+posterior_gap.T)/2)
            assert gap_eigenvalues.min() > -1e-12
            scalar = {}
            for name, g, dg, B in (('rdd',ar[1],dar[1],br[1]),
                                   ('pf',ap[1],pf_derivative[1],bp[1])):
                # Independent scalar clean and noise, both unit variance.
                variance_z = g*g+sigma*sigma
                posterior = 1-g*g/variance_z
                risk_schur = dg*dg+1-(g*dg-sigma)**2/variance_z
                risk_formula = B*B/variance_z
                residuals.extend([abs(posterior-sigma*sigma/variance_z),
                                  abs(risk_schur-risk_formula)])
                scalar[name] = {'g':float(g), 'g_prime':float(dg),
                                'clean_posterior_variance':float(posterior),
                                'velocity_bayes_mse':float(risk_formula),
                                'relative_detail_clean_loss_weight':float(B*B)}
            rows.append({'stage':[a,b], 'fraction':fraction, 't':t,
                         'snr_ratio_rdd_over_pf':float(ratio**2),
                         'gaussian_posterior_gap_eigenvalues':gap_eigenvalues.tolist(),
                         'B_crossing_time':float(crossing), 'scalar':scalar})
    maximum = max(residuals)
    assert maximum < 2e-12, maximum
    middle = [row for row in rows if row['stage']==[.4,.7]]
    assert middle[0]['scalar']['rdd']['velocity_bayes_mse'] < middle[0]['scalar']['pf']['velocity_bayes_mse']
    assert middle[-1]['scalar']['rdd']['velocity_bayes_mse'] > middle[-1]['scalar']['pf']['velocity_bayes_mse']
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'unified_multires/research/rdd_fm_integrator.py',
             root/'unified_multires/umdiff/paths.py']
    result = {
        'scope':'CPU NumPy float64 analytical examples using existing two-band research coefficients; no learned model, author pipeline, GPU, or generation-quality evaluation',
        'assumptions':'Same stage endpoints, physical time, orthogonal P/Q, isotropic independent Gaussian noise; scalar MSE example additionally assumes unit Gaussian clean; matrix example uses a correlated Gaussian clean',
        'posterior_order_scope':'General clean-MMSE ordering follows from the channel-degradation proof in the report, not from these finite Gaussian examples; no uniform ordering of conditional velocity regression risk or learned-model quality',
        'max_identity_residual':maximum, 'cases':rows, 'conditional_actions':actions,
        'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'cases':len(rows), 'max_identity_residual':maximum,
                      'example_stage':middle}, indent=2))


if __name__ == '__main__':
    main()
