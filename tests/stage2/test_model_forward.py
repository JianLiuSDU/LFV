import torch

from lfv.datasets.functional_motion import SyntheticFunctionalMotionDataset
from lfv.models.functional_motion_generation import ThreeTokenHierarchicalDiffusion


def _batch(count=2):
    dataset = SyntheticFunctionalMotionDataset(
        num_samples=count, num_points=32, dino_dim=16
    )
    return {
        key: torch.stack([dataset[index][key] for index in range(count)])
        if torch.is_tensor(dataset[0][key])
        else [dataset[index][key] for index in range(count)]
        for key in dataset[0]
    }


def test_goal_and_trajectory_forward_and_sample():
    batch = _batch()
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        goal_inference_steps=2,
        trajectory_inference_steps=2,
        trajectory_hard_start_token=True,
        trajectory_start_reconstruction_weight=10.0,
        trajectory_start_boundary_weight=1.0,
        trajectory_acceleration_weight=0.1,
    )
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    losses = model.compute_loss(batch, stage="joint")
    assert losses["total"].ndim == 0
    assert torch.isfinite(losses["total"])
    assert torch.isfinite(losses["trajectory_start_boundary"])
    assert torch.isfinite(losses["trajectory_acceleration"])
    samples, encoding = model.sample(
        batch, num_goal_samples=2, num_trajectory_samples=1
    )
    assert encoding.tokens.shape == (2, 3, 32)
    assert samples.goals.shape == (2, 2, 9)
    assert samples.trajectories.shape == (2, 2, 1, 64, 9)
    assert torch.isfinite(samples.trajectories).all()
    torch.testing.assert_close(samples.trajectories[..., 0, :3], torch.zeros_like(samples.trajectories[..., 0, :3]))


def test_sampling_is_reproducible_with_generator_seed():
    batch = _batch(count=1)
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        goal_inference_steps=2,
        trajectory_inference_steps=2,
    ).eval()
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    first, _ = model.sample(
        batch,
        num_goal_samples=2,
        num_trajectory_samples=1,
        generator=torch.Generator().manual_seed(1234),
    )
    second, _ = model.sample(
        batch,
        num_goal_samples=2,
        num_trajectory_samples=1,
        generator=torch.Generator().manual_seed(1234),
    )
    torch.testing.assert_close(first.goals, second.goals)
    torch.testing.assert_close(first.trajectories, second.trajectories)


def test_goal_functional_anchor_and_candidate_scoring():
    batch = _batch(count=2)
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        motion_field_mode="joint",
        goal_relation_conditioning=True,
        goal_candidate_scoring=True,
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        goal_inference_steps=2,
        trajectory_inference_steps=2,
    )
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    losses = model.compute_loss(batch, stage="goal")
    assert torch.isfinite(losses["goal_score"])
    losses["total"].backward()
    assert model.encoder.goal_relation_encoder[0].weight.grad is not None
    samples, encoding = model.sample(
        batch,
        num_goal_samples=3,
        num_trajectory_samples=1,
        generator=torch.Generator().manual_seed(123),
    )
    assert encoding.goal_relation_tokens.shape == (2, 3, 32)
    assert encoding.manipulated_anchor_xyz.shape == (2, 3)
    assert samples.goal_scores.shape == (2, 3)
    assert torch.isfinite(samples.goal_scores).all()


def test_sparsemax_motion_field_is_a_simplex_and_sparse():
    batch = _batch(count=1)
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        motion_field_mode="joint",
        motion_field_normalization="sparsemax",
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
    )
    encoding = model.encode(batch)
    assert encoding.manipulated_motion_field is not None
    field = encoding.manipulated_motion_field
    torch.testing.assert_close(field.sum(dim=1), torch.ones(1))
    assert (field == 0).any()


def test_joint_motion_loss_reaches_both_relevance_heads():
    batch = _batch()
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        motion_field_mode="independent",
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        goal_inference_steps=2,
        trajectory_inference_steps=2,
    )
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    losses = model.compute_loss(batch, stage="joint")
    losses["total"].backward()
    for relation in (
        model.encoder.manipulated_queries_reference,
        model.encoder.reference_queries_manipulated,
    ):
        assert relation.relevance_head is not None
        gradients = [
            parameter.grad
            for parameter in relation.relevance_head.parameters()
            if parameter.requires_grad
        ]
        assert all(gradient is not None for gradient in gradients)
        assert sum(float(gradient.abs().sum()) for gradient in gradients) > 0.0


def test_joint_relation_field_receives_motion_loss_gradients():
    batch = _batch()
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        motion_field_mode="joint",
        motion_field_temperature=0.25,
        motion_field_pair_weight=0.25,
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
        goal_inference_steps=2,
        trajectory_inference_steps=2,
    )
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    encoding = model.encode(batch)
    assert encoding.joint_motion_relation is not None
    encoding.joint_motion_relation.retain_grad()
    losses = model.goal_diffuser.compute_loss(
        encoding.tokens, batch["goal_pose9d"], model.normalizer
    )
    losses["goal_total"].backward()
    assert encoding.joint_motion_relation.grad is not None
    assert float(encoding.joint_motion_relation.grad.abs().sum()) > 0.0
