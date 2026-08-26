import torch

from lfv.models.functional_motion_generation.encoders import BidirectionalSceneEncoder


def test_scene_encoder_shapes_and_permutation_invariance():
    torch.manual_seed(1)
    encoder = BidirectionalSceneEncoder(
        dino_dim=32, hidden_dim=64, dino_projected_dim=32, xyz_projected_dim=32
    ).eval()
    manipulated_points = torch.randn(2, 48, 3)
    manipulated_dino = torch.randn(2, 48, 32)
    reference_points = torch.randn(2, 40, 3)
    reference_dino = torch.randn(2, 40, 32)
    first = encoder(
        manipulated_points,
        manipulated_dino,
        reference_points,
        reference_dino,
        return_debug=True,
    )
    perm_m = torch.randperm(48)
    perm_r = torch.randperm(40)
    second = encoder(
        manipulated_points[:, perm_m],
        manipulated_dino[:, perm_m],
        reference_points[:, perm_r],
        reference_dino[:, perm_r],
    )
    assert first.tokens.shape == (2, 3, 64)
    assert first.attention_manipulated_to_reference.shape == (2, 4, 48, 40)
    assert first.reference_importance.shape == (2, 40)
    torch.testing.assert_close(first.tokens, second.tokens, atol=1e-5, rtol=1e-5)


def test_motion_functional_fields_are_normalized_and_permutation_equivariant():
    torch.manual_seed(7)
    encoder = BidirectionalSceneEncoder(
        dino_dim=16,
        hidden_dim=32,
        dino_projected_dim=16,
        xyz_projected_dim=16,
        num_heads=4,
        dropout=0.0,
        motion_field_mode="independent",
        motion_field_temperature=1.0,
    ).eval()
    manipulated_points = torch.randn(2, 24, 3)
    manipulated_dino = torch.randn(2, 24, 16)
    reference_points = torch.randn(2, 20, 3)
    reference_dino = torch.randn(2, 20, 16)
    first = encoder(
        manipulated_points,
        manipulated_dino,
        reference_points,
        reference_dino,
    )
    assert first.manipulated_motion_field is not None
    assert first.reference_motion_field is not None
    assert first.manipulated_motion_field.shape == (2, 24)
    assert first.reference_motion_field.shape == (2, 20)
    torch.testing.assert_close(
        first.manipulated_motion_field.sum(dim=1), torch.ones(2)
    )
    torch.testing.assert_close(
        first.reference_motion_field.sum(dim=1), torch.ones(2)
    )

    perm_m = torch.randperm(24)
    perm_r = torch.randperm(20)
    second = encoder(
        manipulated_points[:, perm_m],
        manipulated_dino[:, perm_m],
        reference_points[:, perm_r],
        reference_dino[:, perm_r],
    )
    torch.testing.assert_close(first.tokens, second.tokens, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        first.manipulated_motion_field[:, perm_m],
        second.manipulated_motion_field,
        atol=1e-6,
        rtol=1e-6,
    )
    torch.testing.assert_close(
        first.reference_motion_field[:, perm_r],
        second.reference_motion_field,
        atol=1e-6,
        rtol=1e-6,
    )
