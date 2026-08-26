import torch

from lfv.datasets.functional_motion import SyntheticFunctionalMotionDataset
from lfv.diffusion import ExponentialMovingAverage
from lfv.models.functional_motion_generation import ThreeTokenHierarchicalDiffusion
from lfv.training.functional_motion.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_roundtrip(tmp_path):
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=8,
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
    sample = SyntheticFunctionalMotionDataset(
        num_samples=1, num_points=32, dino_dim=8
    )[0]
    batch = {
        key: value[None] if torch.is_tensor(value) else [value]
        for key, value in sample.items()
    }
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    model.eval()
    expected, _ = model.sample(
        batch,
        num_goal_samples=2,
        num_trajectory_samples=1,
        generator=torch.Generator().manual_seed(73),
    )
    ema = ExponentialMovingAverage(model)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "normalizer": model.normalizer.to_dict(),
        },
    )
    payload = load_checkpoint(path)
    restored = ThreeTokenHierarchicalDiffusion(
        dino_dim=8,
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
    restored.load_state_dict(payload["model"])
    restored.normalizer.translation_mean.copy_(model.normalizer.translation_mean)
    restored.normalizer.translation_std.copy_(model.normalizer.translation_std)
    restored.normalizer.fitted.copy_(model.normalizer.fitted)
    restored.eval()
    for first, second in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(first, second)
    reproduced, _ = restored.sample(
        batch,
        num_goal_samples=2,
        num_trajectory_samples=1,
        generator=torch.Generator().manual_seed(73),
    )
    torch.testing.assert_close(expected.goals, reproduced.goals)
    torch.testing.assert_close(expected.trajectories, reproduced.trajectories)


def test_ema_resume_matches_current_model_dtype():
    source = torch.nn.Linear(4, 3).float()
    target = torch.nn.Linear(4, 3).double()
    source_ema = ExponentialMovingAverage(source)
    target_ema = ExponentialMovingAverage(target)
    target_ema.load_state_dict(source_ema.state_dict())
    target_parameters = dict(target.named_parameters())
    for key, shadow in target_ema.shadow.items():
        assert shadow.dtype == target_parameters[key].dtype
        assert shadow.device == target_parameters[key].device
