import numpy as np
import torch

from lfv.datasets.functional_motion import SyntheticFunctionalMotionDataset
from lfv.models.functional_motion_generation import build_model
from lfv.models.functional_motion_generation.canonical_alignment import (
    aggregate_canonical_fields,
    canonical_field_gate,
    pull_target_to_source,
    row_normalize_correspondence,
)
from lfv.models.functional_motion_generation.loading import model_kwargs


def _batch(count=2, num_points=256, dino_dim=16):
    dataset = SyntheticFunctionalMotionDataset(
        num_samples=count, num_points=num_points, dino_dim=dino_dim, seed=17
    )
    items = [dataset[index] for index in range(count)]
    return {
        key: torch.stack([item[key] for item in items])
        if torch.is_tensor(items[0][key])
        else [item[key] for item in items]
        for key in items[0]
    }


def _model(dino_dim=16, hidden_dim=32):
    config = {
        "model": {
            "name": "v7_functional_alignment",
            "motion_field_mode": "local_functional_bottleneck",
            "hidden_dim": hidden_dim,
            "encoder_heads": 4,
            "local_dino_proj_dim": hidden_dim // 2,
            "local_object_xyz_dim": hidden_dim // 4,
            "local_relation_xyz_dim": hidden_dim // 4,
            "field_selector_layers": 2,
            "field_selector_ffn_dim": hidden_dim * 2,
            "field_selector_dropout": 0.0,
            "field_temperature_start": 1.0,
            "field_temperature_end": 0.4,
            "gated_relation_layers": 2,
            "gated_relation_ffn_dim": hidden_dim * 2,
            "functional_pooling_queries": 2,
            "field_target_ratio_start": 0.5,
            "field_target_ratio_end": 0.2,
            "field_knn": 4,
            "field_budget_weight": 0.02,
            "field_smooth_weight": 0.01,
            "field_consistency_weight": 0.0,
            "goal_layers": 1,
            "trajectory_layers": 1,
            "decoder_heads": 4,
            "dropout": 0.0,
            "num_train_timesteps": 10,
            "goal_inference_steps": 2,
            "trajectory_inference_steps": 2,
            "trajectory_hard_start_token": True,
            "trajectory_position_encoding": "discrete_sinusoidal",
            "trajectory_goal_context_layers": 0,
            "trajectory_num_phase_tokens": 0,
        },
        "data": {"dino_dim": dino_dim},
    }
    return build_model(
        "v7_functional_alignment",
        **model_kwargs(config, dino_dim),
    )


def test_v7_context_shape_and_permutation_equivariance():
    torch.manual_seed(3)
    batch = _batch()
    model = _model().eval()
    first = model.encode(batch, return_debug=True)
    assert first.tokens.shape == (2, 5, 32)
    assert first.manipulated_motion_field.shape == (2, 256)
    assert first.reference_motion_field.shape == (2, 256)
    perm_m = torch.randperm(256)
    perm_r = torch.randperm(256)
    permuted = {
        **batch,
        "manipulated_points": batch["manipulated_points"][:, perm_m],
        "manipulated_dino": batch["manipulated_dino"][:, perm_m],
        "manipulated_mask": batch["manipulated_mask"][:, perm_m],
        "reference_points": batch["reference_points"][:, perm_r],
        "reference_dino": batch["reference_dino"][:, perm_r],
        "reference_mask": batch["reference_mask"][:, perm_r],
    }
    second = model.encode(permuted)
    torch.testing.assert_close(first.tokens, second.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        first.manipulated_motion_field[:, perm_m],
        second.manipulated_motion_field,
        atol=1e-5,
        rtol=1e-5,
    )


def test_v7_field_is_a_real_generator_bottleneck_and_has_gradients():
    batch = _batch()
    model = _model()
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    losses = model.compute_loss(batch)
    assert "field_budget" in losses and "field_smoothness" in losses
    losses["total"].backward()
    selector_grad = sum(
        float(p.grad.abs().sum())
        for p in model.encoder.selector.parameters()
        if p.grad is not None
    )
    local_grad = sum(
        float(p.grad.abs().sum())
        for p in model.encoder.manipulated_local_encoder.parameters()
        if p.grad is not None
    )
    relation_grad = sum(
        float(p.grad.abs().sum())
        for p in model.encoder.relation.parameters()
        if p.grad is not None
    )
    assert selector_grad > 0.0
    assert local_grad > 0.0
    assert relation_grad > 0.0
    zeros_m = torch.zeros_like(batch["manipulated_mask"])
    zeros_r = torch.zeros_like(batch["reference_mask"])
    zero_context = model.encode(batch, field_override=(zeros_m, zeros_r))
    assert torch.allclose(zero_context.tokens, torch.zeros_like(zero_context.tokens), atol=1e-6)


def test_v7_interventions_change_context_without_selector_bypass():
    batch = _batch(count=1)
    model = _model().eval()
    learned = model.encode(batch)
    uniform = model.encode(batch, motion_field_intervention="uniform")
    shuffled = model.encode(batch, motion_field_intervention="shuffled")
    assert not torch.allclose(learned.tokens, uniform.tokens)
    assert not torch.allclose(learned.tokens, shuffled.tokens)


def test_source_canonical_alignment_and_field_aggregation():
    source_points = np.arange(12, dtype=np.float32).reshape(4, 3)
    source_dino = np.eye(4, dtype=np.float32)
    target_points = source_points + 0.5
    target_dino = source_dino.copy()
    identity = np.eye(4, dtype=np.float32)
    normalized = row_normalize_correspondence(identity)
    aligned_points, aligned_dino, confidence = pull_target_to_source(
        normalized, target_points, target_dino
    )
    np.testing.assert_allclose(aligned_points, target_points)
    np.testing.assert_allclose(aligned_dino, target_dino)
    np.testing.assert_allclose(confidence, np.ones(4))
    gate = canonical_field_gate(np.array([0.1, 0.8, 0.3, 1.0]), confidence)
    np.testing.assert_allclose(gate, [0.1, 0.8, 0.3, 1.0])
    mean, variance, coverage = aggregate_canonical_fields(
        [np.array([0.0, 1.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.0, 0.0])],
        [identity, identity],
    )
    np.testing.assert_allclose(mean, [0.5, 1.0, 0.0, 0.5])
    np.testing.assert_allclose(variance, [0.25, 0.0, 0.0, 0.25])
    assert np.all(coverage > 0.0)

