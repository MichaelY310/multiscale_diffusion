"""Falsification tests, not a generation or acceleration benchmark."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch


def load_path(filename):
    spec = importlib.util.spec_from_file_location("rdd_flow_audit", filename)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.RDDPath()


def main(args):
    torch.set_num_threads(2)
    torch.manual_seed(314159)
    path = load_path(args.flow)
    clean = torch.randn(1, 2, 2, 8, 8, dtype=torch.float64)
    parent_noise = path.project(torch.randn_like(clean), 2)
    invariance = []
    for t in (.7, .701, .71, .75):
        adjusted_targets, states = [], []
        for _ in range(8):
            eta = torch.randn_like(clean)
            noise = parent_noise + eta - path.project(eta, 2)
            x, w = path.noisy_and_target(clean, noise, torch.tensor([t], dtype=torch.float64), 2)
            adjusted_targets.append(w - (x - path.project(x, 2)) / (1-t))
            states.append(path.project(x, 2))
        stack = torch.stack(adjusted_targets)
        parent_stack = torch.stack(states)
        invariance.append(dict(t=t, adjusted_target_max_difference=float((stack-stack[0]).abs().max()),
                               parent_state_max_difference=float((parent_stack-parent_stack[0]).abs().max())))
    # High-band orthonormal coordinates avoid treating singular Q noise as IID pixels.
    n = 200000
    mu, variance = .4, .01
    h = mu + variance**.5 * torch.randn(n, dtype=torch.float64)
    eps = torch.randn(n, dtype=torch.float64)
    rows = []
    for family in ("rdd", "ordinary_fm"):
        for t in (.7, .701, .71, .75, .8):
            sigma = 1-t
            f, df = path.coefficients(torch.tensor(t, dtype=torch.float64), 2)
            alpha = t-f.item() if family == "rdd" else t
            beta = df.item()-1 if family == "rdd" else -1.
            z = alpha*h + sigma*eps
            target = eps + beta*h
            posterior = mu + alpha*variance/(alpha**2*variance+sigma**2)*(z-alpha*mu)
            multiplier = beta-alpha/sigma
            bayes = z/sigma + multiplier*posterior
            parent_only = z/sigma + multiplier*mu
            gap = multiplier**2*alpha**2*variance**2/(alpha**2*variance+sigma**2)
            empirical_gap = float((parent_only-target).square().mean()-(bayes-target).square().mean())
            rows.append(dict(family=family, t=t, detail_signal_coefficient=alpha,
                             analytic_extra_mse=gap, empirical_extra_mse=empirical_gap,
                             bayes_mse=float((bayes-target).square().mean())))
    a, b = .7, 1.
    z = (1-a)*eps
    df = (b-2*a)/(b-a)
    birth_velocity = z/(1-a)+(df-1)*mu
    one_euler = z-(b-a)*birth_velocity
    result = dict(seed=314159, gaussian_samples=n, conditional_detail_mean=mu,
                  conditional_detail_variance=variance, target_invariance=invariance,
                  scalar_bayes_controls=rows,
                  last_stage_single_euler=dict(mean=float(one_euler.mean()), variance=float(one_euler.var()),
                      expected_mean=a*mu, desired_data_mean=mu, desired_data_variance=variance),
                  conclusions=["Exact birth erases data high-band information from the input.",
                      "Target minus analytic noise is invariant when parent noise and data are fixed.",
                      "That invariance also holds away from birth for fixed data: it is not an exclusivity test.",
                      "Bayes parent-only sufficiency holds only when detail signal coefficient is zero.",
                      "A last-stage one-Euler FM conditional mean destroys detail diversity.",
                      "These are math checks; no speed or perceptual quality was measured."])
    assert invariance[0]["adjusted_target_max_difference"] < 1e-12
    assert abs(one_euler.mean().item()-a*mu) < 1e-12
    assert one_euler.var().item() < 1e-24
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flow", required=True)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
