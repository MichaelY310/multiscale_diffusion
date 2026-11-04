from __future__ import annotations

import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "RDD_DiT_3D"))

from dynamic_sparse_flow import (  # noqa: E402
    fm_data_time_boundaries,
    make_rdd_fm_training_state,
    make_stage_training_state,
    sample_dynamic_sparse_rdd,
)
from dynamic_sparse_model import (  # noqa: E402
    DynamicSparseRDDModel,
    build_neighbor_index,
)
from dynamic_sparse_representation import (  # noqa: E402
    add_empty_parent_groups,
    build_dynamic_sparse_pyramid,
    build_generated_parent_stage,
    dilate_keep_mask,
    project_complete_sibling_detail,
    recover_trellis_features,
)
from calibrate_dynamic_sparse_pruning_replay import (  # noqa: E402
    threshold_for_objective,
)
from train_dynamic_sparse_rdd import reduce_mean, reduce_sum  # noqa: E402
from finetune_dynamic_sparse_occupancy_replay import candidate_labels  # noqa: E402
from compose_dynamic_sparse_thresholds import compose_thresholds  # noqa: E402
from sample_dynamic_sparse_dataset import ResolutionDispatchModel  # noqa: E402


def example_pyramid():
    coords = torch.tensor(
        [
            [0, 0, 0],
            [1, 0, 0],
            [2, 2, 2],
        ],
        dtype=torch.long,
    )
    feats = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]])
    return build_dynamic_sparse_pyramid(
        coords,
        feats,
        fine_resolution=4,
        resolutions=(2, 4),
    )


def test_fixed_eight_empty_representation_and_candidates():
    pyramid = example_pyramid()
    coarse, fine = pyramid.stages
    assert len(coarse.logical_coords) == 2**3
    assert int(coarse.nonempty.sum()) == 2
    # Two active fine children share the first parent, so its mass is 2/8.
    masses = coarse.target_rep[coarse.nonempty, 0].sort().values
    assert torch.allclose(masses, torch.tensor([1 / 8, 2 / 8]))
    # Every non-empty coarse parent contributes exactly eight fine candidates.
    assert len(fine.logical_coords) == 16
    assert int(fine.nonempty.sum()) == 3


def test_transition_is_exact_and_detail_has_zero_sibling_mean():
    fine = example_pyramid().stages[1]
    assert fine.parent_rep is not None and fine.detail is not None
    assert torch.allclose(fine.parent_rep + fine.detail, fine.target_rep)
    projected = project_complete_sibling_detail(fine.logical_coords, fine.detail)
    assert torch.allclose(projected, fine.detail, atol=1e-6)


def test_final_mass_representation_recovers_original_features():
    fine = example_pyramid().stages[1]
    recovered = recover_trellis_features(fine.target_rep[fine.nonempty])
    expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [9.0, 10.0]])
    assert torch.allclose(recovered, expected)


def test_final_recovery_does_not_divide_by_tiny_generated_mass():
    representation = torch.tensor([[1e-8, 2.0, -3.0]])
    recovered = recover_trellis_features(representation)
    assert torch.allclose(recovered, torch.tensor([[2.0, -3.0]]))


def test_empty_parent_augmentation_adds_complete_zero_groups():
    pyramid = example_pyramid()
    stage = pyramid.stages[1]
    augmented = add_empty_parent_groups(
        stage,
        parent_resolution=2,
        fine_resolution=4,
        ratio=1.0,
        generator=torch.Generator().manual_seed(0),
    )
    added = len(augmented.logical_coords) - len(stage.logical_coords)
    assert added == 16
    assert added % 8 == 0
    assert not bool(augmented.nonempty[-added:].any())
    assert torch.count_nonzero(augmented.target_rep[-added:]) == 0


def test_generated_parent_stage_matches_inference_candidates_and_rdd_detail():
    pyramid = example_pyramid()
    generated_parent_coords = torch.tensor([[0, 0, 0], [1, 1, 1]])
    generated_parent_rep = torch.tensor(
        [[0.4, 0.2, -0.1], [0.3, 1.0, 2.0]], dtype=torch.float32
    )
    stage = build_generated_parent_stage(
        pyramid.stages[1],
        parent_coords=generated_parent_coords,
        parent_rep=generated_parent_rep,
        parent_resolution=2,
        fine_resolution=4,
    )
    assert len(stage.logical_coords) == 16
    assert torch.equal(
        stage.parent_rep,
        generated_parent_rep.repeat_interleave(8, dim=0),
    )
    # Generated coarse errors are deliberately not repairable by refinement:
    # every predicted sibling detail group remains exactly zero mean.
    assert torch.allclose(
        stage.detail,
        project_complete_sibling_detail(stage.logical_coords, stage.detail),
        atol=1e-6,
    )


def test_minimal_sparse_model_and_flow_shapes():
    pyramid = example_pyramid()
    stage = pyramid.stages[1]
    training = make_stage_training_state(
        stage,
        stage_index=1,
        num_stages=2,
        local_p=torch.tensor(0.5),
        time_fractions=(0.7, 0.3),
    )
    model = DynamicSparseRDDModel(
        state_channels=3,
        condition_channels=0,
        width=32,
        depth=2,
        position_frequencies=2,
    )
    neighbors = build_neighbor_index(stage.logical_coords, stage.resolution)
    velocity, logits = model(
        training.state,
        stage.logical_coords,
        resolution=stage.resolution,
        global_time=training.global_time,
        neighbor_index=neighbors,
    )
    assert velocity.shape == training.target_velocity.shape
    assert logits.shape == stage.nonempty.shape
    assert torch.isfinite(velocity).all()
    assert torch.isfinite(logits).all()


def test_rdd_fm_300_300_400_uses_2d_noise_to_data_boundaries():
    boundaries = fm_data_time_boundaries((0.3, 0.3, 0.4))
    assert torch.allclose(
        torch.from_numpy(boundaries),
        torch.tensor([0.0, 0.4, 0.7, 1.0], dtype=torch.float64),
    )


def test_rdd_fm_refinement_path_and_velocity_match_2d_formula(monkeypatch):
    stage = example_pyramid().stages[1]
    fixed_noise = torch.arange(stage.target_rep.numel(), dtype=torch.float32).reshape_as(
        stage.target_rep
    ) / 10.0
    monkeypatch.setattr(torch, "randn_like", lambda value: fixed_noise.to(value))
    training = make_rdd_fm_training_state(
        stage,
        stage_index=1,
        num_stages=2,
        local_p=torch.tensor(0.25),
        time_fractions=(0.6, 0.4),
        train_eps=0.0,
    )
    # Coarse interval is [0, .4], fine interval [.4, 1], hence t=.55.
    expected_t = torch.tensor(0.55)
    drift = stage.parent_rep - stage.target_rep
    width = 0.6
    coefficient = expected_t * (1.0 - expected_t) / width
    derivative = (1.0 - 2.0 * expected_t) / width
    expected_state = (
        expected_t * stage.target_rep
        + (1.0 - expected_t) * fixed_noise
        + coefficient * drift
    )
    expected_velocity = stage.target_rep - fixed_noise + derivative * drift
    assert torch.allclose(training.global_time, expected_t)
    assert torch.allclose(training.state, expected_state)
    assert torch.allclose(training.target_velocity, expected_velocity)


def test_single_process_reductions_are_identity():
    value = torch.tensor(3.0)
    assert torch.equal(reduce_sum(value, 1), value)
    assert torch.equal(reduce_mean(value, 1), value)


def test_generated_state_iou_calibration_handles_probability_ties():
    probabilities = torch.tensor([0.9, 0.8, 0.8, 0.1]).numpy()
    labels = torch.tensor([True, True, False, False]).numpy()
    threshold, recall, precision, iou = threshold_for_objective(
        probabilities, labels, "iou", beta=2.0
    )
    # The tied 0.8 examples must enter together: threshold 0.8 predicts 3.
    assert threshold == probabilities[1]
    assert recall == 1.0
    assert abs(precision - 2 / 3) < 1e-8
    assert abs(iou - 2 / 3) < 1e-8


def test_keep_halo_dilates_only_existing_candidates():
    coords = torch.tensor([[0, 0, 0], [0, 0, 1], [2, 2, 2]])
    keep = torch.tensor([True, False, False])
    dilated = dilate_keep_mask(coords, keep, resolution=3, radius=1)
    assert torch.equal(dilated, torch.tensor([True, True, False]))


def test_sampler_preserves_full_candidate_endpoint_states_before_pruning():
    model = DynamicSparseRDDModel(
        state_channels=3,
        condition_channels=0,
        width=32,
        depth=1,
        position_frequencies=2,
    ).eval()
    sample = sample_dynamic_sparse_rdd(
        model,
        resolutions=(2, 4),
        steps=(1, 1),
        time_fractions=(0.7, 0.3),
        probability_thresholds=(1.0, 1.0),
        mass_thresholds=(float("inf"), float("inf")),
        state_channels=3,
        seed=7,
        min_keep=1,
        pruning_mode="and",
        device="cpu",
    )
    assert len(sample.candidate_coords_by_stage) == 2
    assert len(sample.candidate_representations_by_stage) == 2
    # Coarse candidates are the complete 2^3 root grid even though min_keep
    # retains only one parent for refinement.
    assert sample.candidate_coords_by_stage[0].shape == (8, 3)
    assert sample.candidate_representations_by_stage[0].shape == (8, 3)
    assert sample.retained_coords_by_stage[0].shape == (1, 3)
    # That single retained parent expands to all eight children, whose endpoint
    # states must also be available before the second pruning decision.
    assert sample.candidate_coords_by_stage[1].shape == (8, 3)
    assert sample.candidate_representations_by_stage[1].shape == (8, 3)


def test_endpoint_replay_labels_include_empty_and_nonempty_candidates():
    fine_coords = torch.tensor([[0, 0, 0], [3, 3, 3]]).numpy()
    candidates = torch.tensor(
        [[0, 0, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.long
    ).numpy()
    labels = candidate_labels(
        candidates,
        fine_coords,
        fine_resolution=4,
        resolution=2,
    )
    assert labels.tolist() == [True, False, True]


def test_closed_loop_threshold_composition_replaces_only_requested_stage():
    base = {
        "stages": [
            {"resolution": 8, "probability_threshold": 0.9},
            {"resolution": 16, "probability_threshold": 0.8},
            {"resolution": 32, "probability_threshold": 0.7},
        ]
    }
    generated = {
        "objective": "fbeta",
        "fbeta": 2.0,
        "stages": [
            {
                "resolution": 8,
                "probability_threshold": 0.1,
                "candidate_coverage_of_gt": 1.0,
            },
            {
                "resolution": 16,
                "probability_threshold": 0.2,
                "candidate_coverage_of_gt": 0.95,
            },
            {"resolution": 32, "probability_threshold": 0.3},
        ],
    }
    composed = compose_thresholds(base, generated, (16,))
    assert [stage["probability_threshold"] for stage in composed["stages"]] == [
        0.9,
        0.2,
        0.7,
    ]
    assert composed["closed_loop_history"][-1]["replaced_resolutions"] == [16]


def test_resolution_dispatch_uses_the_requested_stage_model():
    calls = []

    class FakeModel:
        def __init__(self, label):
            self.label = label

        def __call__(self, value, *, resolution, **kwargs):
            calls.append((self.label, resolution, value))
            return self.label

    dispatch = ResolutionDispatchModel({8: FakeModel("coarse"), 16: FakeModel("fine")})
    assert dispatch(3, resolution=8) == "coarse"
    assert dispatch(4, resolution=16) == "fine"
    assert calls == [("coarse", 8, 3), ("fine", 16, 4)]
