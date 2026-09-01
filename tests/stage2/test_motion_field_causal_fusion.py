import torch

from lfv.datasets.functional_motion import SyntheticFunctionalMotionDataset
from lfv.models.functional_motion_generation import ThreeTokenHierarchicalDiffusion
from lfv.models.functional_motion_generation.encoders.bidirectional_scene_encoder import _mix_field


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


def test_confidence_fusion_attenuates_disagreeing_prior():
    current = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1]])
    prior = torch.tensor([[0.8, 0.1, 0.1], [0.1, 0.1, 0.8]])
    fused, effective = _mix_field(
        current, prior, 0.5, mode="confidence"
    )
    assert fused.shape == current.shape
    assert torch.allclose(fused.sum(dim=1), torch.ones(2))
    assert effective[0] > effective[1]


def test_causal_probe_and_same_instance_consistency_are_finite():
    batch = _batch()
    batch["field_consistency_group"] = ["cup_0", "cup_0"]
    model = ThreeTokenHierarchicalDiffusion(
        dino_dim=16,
        hidden_dim=32,
        encoder_heads=4,
        motion_field_mode="joint",
        motion_field_bottleneck=True,
        motion_field_causal_weight=0.1,
        motion_field_causal_margin=0.01,
        motion_field_consistency_weight=0.02,
        motion_field_consistency_max_points=16,
        goal_layers=1,
        trajectory_layers=1,
        decoder_heads=4,
        dropout=0.0,
        num_train_timesteps=10,
    )
    model.normalizer.fit_tensors([batch["trajectory_pose9d"]])
    losses = model.compute_loss(batch, stage="joint")
    assert "motion_field_causal" in losses
    assert "motion_field_consistency" in losses
    assert torch.isfinite(losses["motion_field_causal"])
    assert torch.isfinite(losses["motion_field_consistency"])
    assert torch.isfinite(losses["total"])
