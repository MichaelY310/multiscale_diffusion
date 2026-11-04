"""CPU checks against pinned official input masks and block-noise covariance."""
import argparse
import json
import sys
from pathlib import Path
from pyramid_smoke import check_source
from pyramid_compat import block_noise_factor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    hashes = check_source(a.source)
    sys.path.insert(0, str(a.source.resolve()))
    import torch
    from pyramid_dit.flux_modules.modeling_pyramid_flux import PyramidFluxTransformer
    model = PyramidFluxTransformer(in_channels=4, num_layers=0, num_single_layers=0,
                    attention_head_dim=8, num_attention_heads=2, joint_attention_dim=8,
                    pooled_projection_dim=8, axes_dims_rope=[2, 2, 4], use_temporal_causal=True)
    sample = [[torch.zeros(1, 1, 1, 2, 2), torch.zeros(1, 1, 1, 2, 2)]]
    mask = torch.ones(1, 2, dtype=torch.long)
    sdpa = model.merge_input(sample, 2, mask)
    temporal_mask = sdpa[7][0][0, 0]
    assert not bool(temporal_mask[-2, -1])  # history cannot attend current
    assert bool(temporal_mask[-1, -2])      # current can attend history
    model.use_flash_attn = True
    flash = model.merge_input(sample, 2, mask)
    assert flash[7] is None
    assert len(flash[6][0]["indices"]) == 4
    checks = []
    for gamma in [0., .2, 1/3]:
        factor = block_noise_factor(gamma, torch)
        expected = (1+gamma)*torch.eye(4, dtype=torch.float64)-gamma*torch.ones(4, 4, dtype=torch.float64)
        error = float((factor @ factor.T - expected).abs().max())
        assert error < 1e-14
        checks.append({"gamma": gamma, "factor_covariance_max_error": error,
                       "eigenvalues": torch.linalg.eigvalsh(expected).tolist()})
    torch.manual_seed(42)
    noise = torch.randn(100000, 4, dtype=torch.float64) @ block_noise_factor(1/3, torch).T
    empirical = float((noise.T @ noise / len(noise) - expected).abs().max())
    assert empirical < .02
    result = {"source_hashes": hashes, "device": "CPU", "tiny_official_model_mask_check": True,
              "sdpa_mask": temporal_mask.tolist(), "flash_attention_mask": None,
              "official_flash_varlen_causal": False,
              "ar_backend_equivalence": "failed: Flash omits temporal causal mask",
              "noise_factor_checks": checks, "empirical_covariance_max_error": empirical,
              "noise_law_preserved": True, "rng_bitwise_equivalence": False}
    a.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
