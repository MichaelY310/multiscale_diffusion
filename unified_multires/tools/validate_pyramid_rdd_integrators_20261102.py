"""CPU coefficient identities only; no learned model or quality comparison."""
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
        raise FileExistsError('Preserve previous evidence; choose a new output path')
    rows = []
    x = np.array([-.3, 1.2])
    v = np.array([-.7, .4])
    slope = np.array([.2, -.6])
    for a, b in ((.4, .7), (.7, 1.)):
        L = b-a
        for fraction in (.2, .5, .8):
            t = a+fraction*L
            s = t+.15*L
            h, q, r = s-t, 1-t, 1-s
            ratio = r/q
            stage = Stage(a, b)
            at, _, _, bt = stage.coefficients(t)
            ass, _, _, _ = stage.coefficients(s)
            m = (x+q*v)/bt
            corrected = ratio*x+(ass-ratio*at)*m
            euler = x+h*v
            closed = euler+np.array([0., h*h*m[1]/L])
            _, w0, w1 = stage.weights(t, s)
            integral = np.array([quad(
                lambda u: r*stage.coefficients(u)[3][j]/(1-u)**2*(m[j]+(u-t)*slope[j]),
                t, s, epsabs=1e-12, epsrel=1e-12)[0] for j in range(2)])
            # Endpoint-linear path under the same orthogonal two-band basis.
            pf_at = np.array([t, b*(t-a)/L])
            pf_as = np.array([s, b*(s-a)/L])
            pf_derivative = np.array([1., b/L])
            pf_bt = pf_at+q*pf_derivative
            pf_bs = pf_as+r*pf_derivative
            pf_m = (x+q*v)/pf_bt
            pf_corrected = ratio*x+(pf_as-ratio*pf_at)*pf_m
            j1 = h+r*np.log(r/q)
            pf_w0, pf_w1 = (h/q)*pf_bt, j1*pf_bt
            pf_integral = np.array([quad(
                lambda u: r*pf_bt[j]/(1-u)**2*(pf_m[j]+(u-t)*slope[j]),
                t, s, epsabs=1e-12, epsrel=1e-12)[0] for j in range(2)])
            errors = {
                'rdd_curvature_identity': np.max(np.abs(corrected-closed)),
                'rdd_linear_clean_quadrature': np.max(np.abs(w0*m+w1*slope-integral)),
                'pf_zero_order_equals_euler': np.max(np.abs(pf_corrected-euler)),
                'pf_constant_B': np.max(np.abs(pf_bt-pf_bs)),
                'pf_linear_clean_quadrature': np.max(np.abs(pf_w0*pf_m+pf_w1*slope-pf_integral)),
            }
            assert max(errors.values()) < 2e-12, errors
            rows.append({'stage':[a,b], 't':t, 's':s,
                         'rdd_correction':(corrected-euler).tolist(),
                         'max_abs_residuals':{k:float(vv) for k,vv in errors.items()}})
    root = Path(__file__).resolve().parents[2]
    files = [Path(__file__), root/'unified_multires/research/rdd_fm_integrator.py']
    result = {'scope':'CPU NumPy float64 two-band coefficient/quadrature identities only; no model, native scheduler, GPU, or quality test',
              'cases':rows, 'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'cases':len(rows),'max_abs_residual':max(max(row['max_abs_residuals'].values()) for row in rows)}))


if __name__ == '__main__':
    main()
