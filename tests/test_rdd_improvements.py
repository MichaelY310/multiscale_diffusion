import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(1, str(ROOT / "DiT"))

from diffusion.rdd_improvements import (  # noqa: E402
    GuidanceConfig,
    apply_autoguidance_to_model_output,
    apply_cfg_to_model_output,
    diffusion_loss_weights,
    pool_teacher_tokens,
    representation_alignment_loss,
    sample_fm_time,
    weighted_diffusion_loss,
)
from models_multires import DiTMultiRes  # noqa: E402
from diffusion.repa import load_state_dict_with_optional_repa  # noqa: E402


class _EpsilonDiffusion:
    class _MeanType:
        name = "EPSILON"

    model_mean_type = _MeanType()
    alphas_cumprod = np.array([0.9, 0.5, 0.1], dtype=np.float64)


def test_min_snr_and_p2_weight_only_mse():
    diffusion = _EpsilonDiffusion()
    timesteps = torch.tensor([0, 1, 2])
    min_snr = diffusion_loss_weights(diffusion, timesteps, "min-snr", min_snr_gamma=5)
    assert torch.allclose(min_snr, torch.tensor([5 / 9, 1.0, 1.0]), atol=1e-6)

    p2 = diffusion_loss_weights(diffusion, timesteps, "p2", p2_gamma=1, p2_k=1)
    expected_p2 = 1 / (1 + torch.tensor([9.0, 1.0, 1 / 9]))
    assert torch.allclose(p2, expected_p2, atol=1e-6)

    terms = {
        "mse": torch.ones(3),
        "vb": torch.full((3,), 2.0),
        "loss": torch.full((3,), 3.0),
    }
    loss, weights = weighted_diffusion_loss(
        terms, diffusion, timesteps, "min-snr", min_snr_gamma=5
    )
    assert torch.allclose(loss, weights + 2.0)


def test_guidance_interval_and_output_composition():
    config = GuidanceConfig(scale=2.0, low=0.2, high=0.8)
    scales = config.scales(torch.tensor([0.1, 0.5, 0.9]))
    assert torch.equal(scales, torch.tensor([1.0, 2.0, 1.0]))

    cond = torch.full((2, 5, 1, 1), 3.0)
    uncond = torch.full((2, 5, 1, 1), 1.0)
    output = apply_cfg_to_model_output(
        torch.cat([cond, uncond]), torch.tensor([2.0, 1.0]), guided_channels=3
    )
    first = output[:2]
    assert torch.equal(first[0, :3], torch.full((3, 1, 1), 5.0))
    assert torch.equal(first[1, :3], torch.full((3, 1, 1), 3.0))
    assert torch.equal(first[:, 3:], cond[:, 3:])

    good = torch.full((2, 6, 1, 1), 3.0)
    bad = torch.full((2, 6, 1, 1), 1.0)
    auto = apply_autoguidance_to_model_output(
        good, bad, torch.tensor([2.0, 1.0]), guided_channels=4
    )
    assert torch.equal(auto[0, :4], torch.full((4, 1, 1), 5.0))
    assert torch.equal(auto[:, 4:], good[:, 4:])


def test_stage_local_fm_time_and_teacher_pooling():
    uniform = sample_fm_time(128, torch.device("cpu"), sampler="uniform")
    logit = sample_fm_time(128, torch.device("cpu"), sampler="logit-normal")
    assert bool(((uniform > 0) & (uniform < 1)).all())
    assert bool(((logit > 0) & (logit < 1)).all())

    teacher = torch.randn(2, 16 * 16, 8)
    pooled = pool_teacher_tokens(teacher, 4 * 4)
    assert pooled.shape == (2, 16, 8)
    assert representation_alignment_loss(pooled, pooled).shape == (2,)
    assert torch.allclose(
        representation_alignment_loss(pooled, pooled),
        torch.full((2,), -1.0),
        atol=1e-5,
    )


def test_multires_repa_is_opt_in_and_legacy_state_dict_stays_strict():
    legacy = DiTMultiRes(
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        base_size=4,
    )
    assert not any(key.startswith("repa_projector") for key in legacy.state_dict())
    clone = DiTMultiRes(
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        base_size=4,
    )
    clone.load_state_dict(legacy.state_dict(), strict=True)

    repa = DiTMultiRes(
        patch_size=2,
        hidden_size=32,
        depth=2,
        num_heads=4,
        num_classes=10,
        base_size=4,
        repa_dim=8,
        repa_depth=1,
        repa_projector_dim=16,
    )
    x = torch.randn(2, 4, 8, 8)
    t = torch.randint(0, 1000, (2,))
    y = torch.randint(0, 10, (2,))
    output, features = repa(x, t, y, return_features=True)
    assert output.shape == (2, 8, 8, 8)
    assert features.shape == (2, 16, 8)

    load_state_dict_with_optional_repa(
        repa, legacy.state_dict(), allow_missing_repa=True
    )
    broken = dict(legacy.state_dict())
    broken.pop("final_layer.linear.weight")
    try:
        load_state_dict_with_optional_repa(
            repa, broken, allow_missing_repa=True
        )
    except RuntimeError as exc:
        assert "outside the optional REPA projector" in str(exc)
    else:
        raise AssertionError("A non-REPA checkpoint mismatch was silently accepted")


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().items()
        if name.startswith("test_") and callable(value)
    )
    for name, test in tests:
        test()
        print(f"PASS {name}")
    print(f"Passed {len(tests)} RDD improvement tests")
