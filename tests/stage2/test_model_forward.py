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
