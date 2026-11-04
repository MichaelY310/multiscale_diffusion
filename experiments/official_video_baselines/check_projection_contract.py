"""Independent projection algebra, covariance and odd-frame shape checks on CPU."""
import hashlib
import json
from pathlib import Path

from fastvideo_cpu_import import bootstrap


def main() -> None:
    bootstrap(Path('/scr/yuheng/bugswarm/official_video_baselines_20261004/reference/FastVideo-37dd31d5'))
    import torch
    from spatiotemporal_rdd import FirstFrameProjectionPath, FirstFrameDriftFreePath
    torch.set_num_threads(4)
    torch.manual_seed(61)
    checks = []
    for name, factors in {
            'spatial': ((4, 1), (2, 1), (1, 1)),
            'temporal': ((1, 4), (1, 2), (1, 1)),
            'joint': ((4, 4), (2, 2), (1, 1))
    }.items():
        path = FirstFrameProjectionPath(factors=factors)
        full = torch.randn(2, 3, 21, 16, 32, dtype=torch.float64)
        for stage, (spatial, temporal) in enumerate(factors):
            clean = path.stage_clean(full, stage)
            assert tuple(clean.shape[2:]) == (1 + 20 // temporal, 16 // spatial, 32 // spatial)
            projection = path.project(clean, stage)
            torch.testing.assert_close(path.project(projection, stage), projection, atol=1e-14, rtol=1e-14)
            if stage:
                # Conditional expectation is orthogonal: residual averages vanish.
                old = factors[stage - 1]
                residual = path.pool(clean - projection, old[0] // spatial, old[1] // temporal)
                assert residual.abs().max() < 1e-14
                # Neither first-frame value nor tail may leak across the split.
                first_only = torch.zeros_like(clean)
                first_only[:, :, :1] = 7
                assert path.project(first_only, stage)[:, :, 1:].count_nonzero() == 0
            a, b = path.boundaries[stage:stage + 2]
            t = torch.full((2, ), a + .37 * (b - a), dtype=torch.float64)
            noise = torch.randn_like(clean)
            x, w = path.noisy_and_target(clean, noise, t, stage)
            torch.testing.assert_close(path.clean_from_velocity(x, w, t, stage), clean, atol=1e-12, rtol=1e-12)
            delta = 1e-6
            plus = path.noisy_and_target(clean, noise, t + delta, stage)[0]
            minus = path.noisy_and_target(clean, noise, t - delta, stage)[0]
            torch.testing.assert_close((plus - minus) / (2 * delta), -w, atol=1e-8, rtol=1e-8)
            if stage and projection.sub(clean).abs().max() > 1e-3:
                assert (x - (1 - t[:, None, None, None, None]) * w - clean).abs().max() > .01
            baseline = FirstFrameDriftFreePath(factors=factors)
            plain_x, plain_w = baseline.noisy_and_target(clean, noise, t, stage)
            torch.testing.assert_close(plain_w, noise - clean)
            torch.testing.assert_close(baseline.clean_from_velocity(plain_x, plain_w, t, stage), clean)
            if stage < 2:
                next_x = path.transition(clean, full.shape, stage + 1, torch.Generator().manual_seed(51))
                s2, t2 = factors[stage + 1]
                assert tuple(next_x.shape[2:]) == (1 + 20 // t2, 16 // s2, 32 // s2)
            # Independent IID draws check unit variance in both temporal regions.
            normalized = path.noise_at_stage(torch.randn(2000, 1, 21, 4, 4), stage)
            variances = [float(normalized[:, :, :1].var()), float(normalized[:, :, 1:].var())]
            assert all(abs(v - 1) < .08 for v in variances)
            checks.append(dict(projection=name, stage=stage, shape=list(clean.shape), noise_variances=variances))
    assert not torch.cuda.is_initialized()
    work = Path(__file__).resolve().parents[3]
    (work / 'PROJECTION_ALGEBRA_CPU_20261006.json').write_text(
        json.dumps(dict(status='passed',
                        checks=checks,
                        cuda_initialized=False,
                        checks_include=[
                            'idempotence', 'orthogonal residual', 'first-frame isolation', 'clean inverse',
                            'velocity finite difference', 'separately defined drift-free path', 'transition shape',
                            'noise covariance'
                        ],
                        source_sha256=hashlib.sha256(
                            (Path(__file__).parent / 'spatiotemporal_rdd.py').read_bytes()).hexdigest()),
                   indent=2))


if __name__ == '__main__':
    main()
