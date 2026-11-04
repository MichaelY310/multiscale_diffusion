from types import SimpleNamespace

import torch

from rdd_video.attention_analysis.fastwan_native_attention import (
    FastWanNativeAttentionController,
)


def make_controller() -> FastWanNativeAttentionController:
    controller = FastWanNativeAttentionController.__new__(
        FastWanNativeAttentionController
    )
    controller.model = SimpleNamespace(patch_size=(1, 2, 2), blocks=[None])
    controller.cross_stage_reuse_layers = frozenset({0})
    controller.cross_stage_reuse_layers_by_stage = {}
    controller.cross_stage_reuse_stages = frozenset({1})
    controller.cross_stage_reuse_scale = 0.5
    controller.cross_stage_reuse_scale_by_stage = {}
    controller.cross_stage_reuse_layer_coefficients = {}
    controller.cross_stage_reuse_layer_coefficients_by_stage = {}
    controller.cross_stage_reuse_precompose = False
    controller.cross_stage_reuse_resize = "nearest"
    controller.current_stage = 1
    controller.current_shape = (2, 4, 4)
    controller.reuse_scope = "backend"
    controller.reuse_every = 1
    controller.attention_mode = "vsa"
    controller.routes = {}
    controller.stats = {
        "cross_stage_block_reuse_hits": 0,
        "cross_stage_block_reuse_captures": 0,
        "cross_stage_multiscale_predictor_hits": 0,
        "cross_stage_fine_highpass_hits": 0,
        "cross_stage_precompose_captures": 0,
        "cross_stage_precompose_hits": 0,
        "route_reuse_aliases": 0,
    }
    controller.cross_stage_precomposed_cache = {}
    return controller


def test_nearest_resize_preserves_t_h_w_layout() -> None:
    controller = make_controller()
    source = torch.arange(2.0).view(1, 2, 1)
    resized = controller._resize_sequence_between_stages(
        source, source_shape=(2, 2, 2), target_shape=(2, 4, 4)
    )
    assert resized.shape == (1, 8, 1)
    assert torch.equal(resized[0, :4, 0], torch.zeros(4))
    assert torch.equal(resized[0, 4:, 0], torch.ones(4))


def test_cross_stage_reuse_adds_scaled_residual_and_cascades_cache() -> None:
    controller = make_controller()
    source_residual = torch.ones(1, 2, 3)
    controller.cross_stage_residual_cache = {
        (0, 0): ((2, 2, 2), source_residual)
    }

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fine block should have been skipped")

    controller.original_block_forwards = [fail_if_called]
    hidden = torch.zeros(1, 8, 3)

    output = controller._dispatch_block(
        0,
        None,
        hidden,
        torch.empty(0),
        torch.empty(0),
        (torch.empty(0), torch.empty(0)),
        8,
    )

    assert torch.equal(output, torch.full_like(hidden, 0.5))
    assert controller.stats["cross_stage_block_reuse_hits"] == 1
    cached_shape, cached_residual = controller.cross_stage_residual_cache[(1, 0)]
    assert cached_shape == (2, 4, 4)
    assert torch.equal(cached_residual, torch.full_like(hidden, 0.5))


def test_multiscale_predictor_uses_two_stage_residuals() -> None:
    controller = make_controller()
    controller.current_stage = 2
    controller.cross_stage_reuse_stages = frozenset({2})
    controller.cross_stage_reuse_layer_coefficients = {0: (0.6, -0.2, 0.0)}
    controller.cross_stage_residual_cache = {
        (0, 0): ((2, 2, 2), torch.ones(1, 2, 3)),
        (1, 0): ((2, 4, 4), torch.full((1, 8, 3), 3.0)),
    }
    controller.current_shape = (2, 8, 8)

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("fine block should have been skipped")

    controller.original_block_forwards = [fail_if_called]
    hidden = torch.zeros(1, 32, 3)
    output = controller._dispatch_block(
        0,
        None,
        hidden,
        torch.empty(0),
        torch.empty(0),
        (torch.empty(0), torch.empty(0)),
        32,
    )
    # 0.6*M - 0.2*(M-C) = 0.6*3 - 0.2*(3-1) = 1.4.
    assert torch.allclose(output, torch.full_like(hidden, 1.4))
    assert controller.stats["cross_stage_multiscale_predictor_hits"] == 1


def test_precomposed_multiscale_predictor_matches_fine_stage_formula() -> None:
    controller = make_controller()
    controller.cross_stage_reuse_precompose = True
    controller.cross_stage_reuse_stages = frozenset({2})
    controller.cross_stage_reuse_layer_coefficients = {0: (0.6, -0.2, 0.0)}
    controller.current_stage = 1
    controller.current_shape = (2, 4, 4)
    controller.cross_stage_residual_cache = {
        (0, 0): ((2, 2, 2), torch.ones(1, 2, 3)),
        (1, 0): ((2, 4, 4), torch.full((1, 8, 3), 3.0)),
    }
    controller._maybe_precompose_cross_stage_residual(
        0, (2, 4, 4), controller.cross_stage_residual_cache[(1, 0)][1]
    )
    assert controller.stats["cross_stage_precompose_captures"] == 1

    controller.current_stage = 2
    controller.current_shape = (2, 8, 8)
    controller.original_block_forwards = [
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("fine block should have been skipped")
        )
    ]
    hidden = torch.zeros(1, 32, 3)
    output = controller._dispatch_block(
        0,
        None,
        hidden,
        torch.empty(0),
        torch.empty(0),
        (torch.empty(0), torch.empty(0)),
        32,
    )
    assert torch.allclose(output, torch.full_like(hidden, 1.4))
    assert controller.stats["cross_stage_precompose_hits"] == 1


def test_stage_specific_layers_and_coefficients_do_not_leak_between_stages() -> None:
    controller = make_controller()
    controller.cross_stage_reuse_layers = frozenset()
    controller.cross_stage_reuse_layers_by_stage = {
        1: frozenset({0}),
        2: frozenset(),
    }
    controller.cross_stage_reuse_stages = frozenset({1, 2})
    controller.cross_stage_reuse_layer_coefficients_by_stage = {
        1: {0: (0.25, 0.0, 0.0)}
    }
    controller.current_stage = 1
    controller.current_shape = (2, 4, 4)
    controller.cross_stage_residual_cache = {
        (0, 0): ((2, 2, 2), torch.ones(1, 2, 3))
    }
    controller.original_block_forwards = [
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("middle block should have been skipped")
        )
    ]
    hidden = torch.zeros(1, 8, 3)
    output = controller._dispatch_block(
        0,
        None,
        hidden,
        torch.empty(0),
        torch.empty(0),
        (torch.empty(0), torch.empty(0)),
        8,
    )
    assert torch.allclose(output, torch.full_like(hidden, 0.25))
    assert 0 not in controller._reuse_layers_for_stage(2)


if __name__ == "__main__":
    test_nearest_resize_preserves_t_h_w_layout()
    test_cross_stage_reuse_adds_scaled_residual_and_cascades_cache()
    test_multiscale_predictor_uses_two_stage_residuals()
    test_precomposed_multiscale_predictor_matches_fine_stage_formula()
    test_stage_specific_layers_and_coefficients_do_not_leak_between_stages()
    print("CSBR tests passed")
