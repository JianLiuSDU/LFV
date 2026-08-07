import pytest
import torch

from lfv.models.functional_motion_generation.blocks import SinusoidalEmbedding
from lfv.models.functional_motion_generation.loading import model_kwargs
from lfv.models.functional_motion_generation.trajectory import TrajectoryDecoder


@pytest.mark.parametrize(
    ("use_hard_start_token", "expected"),
    [
        (True, [0.0, 1.0, 2.0, 3.0]),
        (False, [1.0, 2.0, 3.0]),
    ],
)
def test_discrete_position_encoding_uses_absolute_frame_indices(
    use_hard_start_token, expected
):
    decoder = TrajectoryDecoder(
        hidden_dim=8,
        num_layers=0,
        num_heads=2,
        dropout=0.0,
        horizon_without_start=3,
        use_hard_start_token=use_hard_start_token,
        position_encoding="discrete_sinusoidal",
    )
    positions = decoder._frame_positions(device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(positions, torch.tensor(expected))


def test_discrete_sinusoidal_positions_do_not_collapse_to_one_dimension():
    positions = torch.arange(64, dtype=torch.float32)
    embeddings = SinusoidalEmbedding(128)(positions)
    singular_values = torch.linalg.svdvals(embeddings - embeddings.mean(dim=0))
    participation_rank = singular_values.sum().square() / singular_values.square().sum()
    adjacent_distance = torch.linalg.vector_norm(embeddings[1:] - embeddings[:-1], dim=-1)

    assert participation_rank.item() > 10.0
    assert adjacent_distance.median().item() > 1.0


def test_legacy_position_encoding_is_explicit_and_reproducible():
    decoder = TrajectoryDecoder(
        hidden_dim=8,
        num_layers=0,
        num_heads=2,
        dropout=0.0,
        horizon_without_start=3,
        use_hard_start_token=True,
        position_encoding="legacy_normalized_sinusoidal",
    )
    positions = decoder._frame_positions(device=torch.device("cpu"), dtype=torch.float32)
    torch.testing.assert_close(positions, torch.linspace(0.0, 1.0, 4))


def test_unknown_position_encoding_fails_fast():
    with pytest.raises(ValueError, match="Unknown trajectory position encoding"):
        TrajectoryDecoder(position_encoding="unknown")


def test_checkpoint_config_without_position_version_uses_legacy_mode():
    kwargs = model_kwargs({"model": {}}, dino_dim=16)
    assert kwargs["trajectory_position_encoding"] == "legacy_normalized_sinusoidal"


def test_explicit_checkpoint_position_version_is_preserved():
    kwargs = model_kwargs(
        {"model": {"trajectory_position_encoding": "discrete_sinusoidal"}},
        dino_dim=16,
    )
    assert kwargs["trajectory_position_encoding"] == "discrete_sinusoidal"
