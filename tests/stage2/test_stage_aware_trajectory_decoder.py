import pytest
import torch

from lfv.models.functional_motion_generation.trajectory import TrajectoryDecoder


@pytest.mark.parametrize(
    ("context_layers", "phase_tokens", "attention_mode", "gating"),
    [
        (2, 0, "full", False),
        (2, 4, "full", False),
        (2, 4, "alternating_local_global", True),
    ],
)
def test_stage_aware_decoder_variants_preserve_shape(
    context_layers, phase_tokens, attention_mode, gating
) -> None:
    decoder = TrajectoryDecoder(
        hidden_dim=32,
        num_layers=4,
        num_heads=4,
        dropout=0.0,
        horizon_without_start=7,
        use_hard_start_token=True,
        goal_context_layers=context_layers,
        num_phase_tokens=phase_tokens,
        temporal_attention_mode=attention_mode,
        residual_gating=gating,
    )
    output = decoder(
        torch.randn(2, 7, 9),
        torch.tensor([1, 7]),
        torch.randn(2, 3, 32),
        torch.randn(2, 9),
        normalized_start=torch.randn(2, 9),
    )
    assert output.shape == (2, 7, 9)
    assert torch.isfinite(output).all()
    if gating:
        assert decoder.blocks[0].conv_scale is not None
        assert decoder.blocks[0].attention_mode == "local"
        assert decoder.blocks[1].attention_mode == "full"


def test_phase_bias_is_ordered_and_peaks_near_phase_centres() -> None:
    decoder = TrajectoryDecoder(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        num_phase_tokens=4,
    )
    bias = decoder._phase_attention_bias(
        64, device=torch.device("cpu"), dtype=torch.float32
    )
    peaks = bias.argmax(dim=1)
    assert torch.all(peaks[1:] >= peaks[:-1])
    assert peaks[0].item() == 0
    assert peaks[-1].item() == 3


def test_context_mixer_gate_starts_as_small_residual_refinement() -> None:
    decoder = TrajectoryDecoder(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        goal_context_layers=2,
        goal_context_residual_gating=True,
        residual_gate_init=0.1,
    )
    gate = decoder.context_mixer.residual_gate
    assert gate is not None
    torch.testing.assert_close(gate, torch.full_like(gate, 0.1))


def test_phase_branch_can_be_gated_without_scaling_existing_branches() -> None:
    decoder = TrajectoryDecoder(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        num_phase_tokens=4,
        phase_residual_gating=True,
        residual_gating=False,
        residual_gate_init=0.1,
    )
    block = decoder.blocks[0]
    assert block.phase_scale is not None
    assert block.conv_scale is None
    assert block.self_scale is None
    assert block.cross_scale is None
    assert block.ffn_scale is None


def test_local_attention_mask_only_exposes_neighbourhood() -> None:
    decoder = TrajectoryDecoder(
        hidden_dim=16,
        num_layers=1,
        num_heads=4,
        temporal_attention_mode="alternating_local_global",
        temporal_local_window=3,
    )
    mask = decoder.blocks[0]._self_attention_mask(torch.zeros(1, 5, 16))
    assert mask is not None
    assert not mask[2, 1] and not mask[2, 2] and not mask[2, 3]
    assert mask[2, 0] and mask[2, 4]
