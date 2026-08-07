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
