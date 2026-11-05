"""CPU audit of fresh kernels and the actual legacy DDPM / FM noise clocks.

No model/data downloads or learned-network quality claims. Run from any cwd:
python unified_multires/tools/audit_rdd_fresh_semantics.py --output report.json
"""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'unified_multires')]

import numpy as np
import torch
from diffusion import create_diffusion_with_drift
from diffusion.flow_matching_with_drift import FlowMatchingWithDrift, upsample_with_noise_fm
from diffusion.gaussian_diffusion_with_drift import _extract_into_tensor
from umdiff.boundaries import RDDFresh
from umdiff.rdd_native import NativeRDDStages


def audit():
    torch.set_num_threads(2)
    torch.manual_seed(431)
    # Execute the actual legacy function without importing its full launcher.
    path = ROOT / '2d/sample_fid_unified.py'
    function = next(n for n in ast.parse(path.read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == '_redraw_upsample')
    namespace = dict(torch=torch, _extract_into_tensor=_extract_into_tensor)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
    legacy_fresh = namespace['_redraw_upsample']
    kernels = []
    for coarse, time in [(8, .3), (16, .6)]:
        clean = torch.randn(2, 4, coarse, coarse)
        # Equal coefficients isolate the boundary operation. These artificial
        # arrays are NOT asserted to be a VP/DDPM probability schedule.
        coeffs = SimpleNamespace(sqrt_alphas_cumprod=np.array([time]),
                                 sqrt_one_minus_alphas_cumprod=np.array([1-time]))
        torch.manual_seed(991)
        ddpm, noise = legacy_fresh(clean, torch.zeros(2, dtype=torch.long), coeffs, coarse*2)
        ddpm_rng = torch.get_rng_state()
        torch.manual_seed(991)
        fm = upsample_with_noise_fm(clean, time, coarse*2)
        fm_rng = torch.get_rng_state()
        component = RDDFresh(2)(torch.randn_like(clean), clean=clean,
                               alpha=time, sigma=1-time, noise=noise).state
        error = max((fm-ddpm).abs().max().item(), (fm-component).abs().max().item())
        exact = all(torch.equal(fm.contiguous().view(torch.uint8),
                                x.contiguous().view(torch.uint8)) for x in (ddpm, component))
        assert error == 0 and exact and torch.equal(ddpm_rng, fm_rng)
        kernels.append(dict(coarse_grid=coarse, alpha=time, sigma=1-time,
                            max_abs_error=error, byte_exact=exact, rng_equal=True))

    inverses = []
    entries = []
    for stage, (start, end) in enumerate(zip([0., .3, .6], [.3, .6, 1.])):
        fm = FlowMatchingWithDrift(c=None if stage == 0 else stage,
                                   s_k=start, s_k_plus_1=end)
        clean = torch.randn(3, 4, 8, 8, dtype=torch.float64)
        noise = torch.randn_like(clean)
        times = torch.tensor([start, (start+end)/2, end], dtype=clean.dtype)
        drift = None if stage == 0 else fm.compute_drift(clean)
        state = fm.q_sample(clean, times, noise, drift)
        target = fm.compute_velocity_target(clean, noise, times, drift)
        recovered = fm.predict_x1_from_velocity(state, target, times)
        error = (recovered-clean).abs().max().item()
        assert error < 1e-12
        inverses.append(dict(stage=stage, times=times.tolist(), oracle_clean_max_abs_error=error))
        if stage:
            time = torch.full((3,), start, dtype=clean.dtype)
            expected = start*fm.upsample(fm.downsample(clean, 2), 2)+(1-start)*noise
            actual = fm.q_sample(clean, time, noise, drift)
            entry_error = (actual-expected).abs().max().item()
            assert entry_error < 1e-12
            entries.append(dict(stage=stage, entry_formula_max_abs_error=entry_error))

    config_path = ROOT / 'unified_multires/configs/rdd_reproduction/rdd_ddpm_xl_300_300_400_ddpm_cfg1p0_fresh.json'
    sampler = NativeRDDStages(**json.loads(config_path.read_text())['sampler']['args'])
    clocks = []
    for index in reversed(range(3)):
        diffusion = sampler.diffusions[index]
        train = create_diffusion_with_drift('', c=diffusion.c, T_c=diffusion.T_c,
                    T_c_plus_1=diffusion.T_c_plus_1, sample_scale=diffusion.sample_scale)
        selection = [t-diffusion.T_c for t in diffusion.timestep_map]
        train_error = float(np.max(np.abs(train.alphas_cumprod[selection]-diffusion.alphas_cumprod)))
        assert train_error < 1e-12
        fm_start, fm_end = [(0., .3), (.3, .6), (.6, 1.)][2-index]
        clocks.append(dict(grid=sampler.resolutions[index],
            ddpm_model_times=[diffusion.timestep_map[-1], diffusion.timestep_map[0]],
            ddpm_entry_alpha=float(diffusion.sqrt_alphas_cumprod[-1]),
            ddpm_entry_sigma=float(diffusion.sqrt_one_minus_alphas_cumprod[-1]),
            ddpm_exit_alpha=float(diffusion.sqrt_alphas_cumprod[0]),
            ddpm_exit_sigma=float(diffusion.sqrt_one_minus_alphas_cumprod[0]),
            training_sampling_alpha_max_abs_error=train_error,
            fm_times=[fm_start, fm_end], fm_exit_alpha=fm_end, fm_exit_sigma=1-fm_end))
    files = [path, config_path, ROOT/'diffusion/flow_matching_with_drift.py',
             ROOT/'diffusion/respace.py', ROOT/'diffusion/gaussian_diffusion_with_drift.py',
             ROOT/'unified_multires/umdiff/rdd_native.py',
             ROOT/'unified_multires/umdiff/rdd_fm_native.py', Path(__file__)]
    return dict(status='passed', device='cpu', equal_coefficient_boundary_checks=kernels,
        fm_clean_oracle_checks=inverses, fm_training_entry_checks=entries, actual_clocks=clocks,
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope='Boundary kernel algebra and true coefficients; no learned weights, GPU, complete sampling or FID rerun.',
        conclusion='Same fresh operation for equal coefficients; actual DDPM/FM noise clocks differ. Legacy DDPM cumulative alpha resets per stage in training and sampling.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))
