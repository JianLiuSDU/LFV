import torch

from lfv.geometry import matrix_to_rotation_6d, rotation_6d_to_matrix


def test_rotation6d_roundtrip_and_gradient():
    matrix = torch.eye(3).repeat(4, 1, 1)
    rotation = matrix_to_rotation_6d(matrix).requires_grad_(True)
    recovered = rotation_6d_to_matrix(rotation)
    torch.testing.assert_close(recovered, matrix, atol=1e-6, rtol=1e-6)
    recovered.square().sum().backward()
    assert rotation.grad is not None
    assert torch.isfinite(rotation.grad).all()
