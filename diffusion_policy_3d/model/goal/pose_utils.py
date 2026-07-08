import torch
import torch.nn.functional as F


def _check_last_dim(x, dim, name):
    if x.shape[-1] != dim:
        raise ValueError(f"{name} expected last dim {dim}, got shape {tuple(x.shape)}")


def normalize_quat(q, eps=1e-8):
    _check_last_dim(q, 4, "q")
    q = F.normalize(q, dim=-1, eps=eps)
    return q


def quat_to_matrix(q):
    _check_last_dim(q, 4, "q")
    q = normalize_quat(q)
    x, y, z, w = q.unbind(dim=-1)
    two_s = 2.0 / (q * q).sum(dim=-1).clamp_min(1e-8)

    xx = two_s * x * x
    yy = two_s * y * y
    zz = two_s * z * z
    xy = two_s * x * y
    xz = two_s * x * z
    yz = two_s * y * z
    xw = two_s * x * w
    yw = two_s * y * w
    zw = two_s * z * w

    row0 = torch.stack((1 - yy - zz, xy - zw, xz + yw), dim=-1)
    row1 = torch.stack((xy + zw, 1 - xx - zz, yz - xw), dim=-1)
    row2 = torch.stack((xz - yw, yz + xw, 1 - xx - yy), dim=-1)
    return torch.stack((row0, row1, row2), dim=-2)


def matrix_to_quat(R):
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R expected shape [...,3,3], got {tuple(R.shape)}")

    m00, m01, m02 = R[..., 0, 0], R[..., 0, 1], R[..., 0, 2]
    m10, m11, m12 = R[..., 1, 0], R[..., 1, 1], R[..., 1, 2]
    m20, m21, m22 = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]

    qw = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))
    qx = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0))
    qy = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0))
    qz = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0))

    qx = torch.copysign(qx, m21 - m12)
    qy = torch.copysign(qy, m02 - m20)
    qz = torch.copysign(qz, m10 - m01)
    q = torch.stack((qx, qy, qz, qw), dim=-1)
    return normalize_quat(q)


def rot6d_to_matrix(x):
    _check_last_dim(x, 6, "rot6d")
    a1 = x[..., 0:3]
    a2 = x[..., 3:6]
    b1 = F.normalize(a1, dim=-1, eps=1e-8)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(R):
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"R expected shape [...,3,3], got {tuple(R.shape)}")
    return R[..., :, :2].transpose(-1, -2).reshape(*R.shape[:-2], 6)


def pose7d_to_matrix(pose7d):
    _check_last_dim(pose7d, 7, "pose7d")
    R = quat_to_matrix(pose7d[..., 3:7])
    T = torch.zeros(*pose7d.shape[:-1], 4, 4, dtype=pose7d.dtype, device=pose7d.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = pose7d[..., :3]
    T[..., 3, 3] = 1
    return T


def pose9d_to_matrix(pose9d):
    _check_last_dim(pose9d, 9, "pose9d")
    R = rot6d_to_matrix(pose9d[..., 3:9])
    T = torch.zeros(*pose9d.shape[:-1], 4, 4, dtype=pose9d.dtype, device=pose9d.device)
    T[..., :3, :3] = R
    T[..., :3, 3] = pose9d[..., :3]
    T[..., 3, 3] = 1
    return T


def matrix_to_pose7d(T):
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"T expected shape [...,4,4], got {tuple(T.shape)}")
    quat = matrix_to_quat(T[..., :3, :3])
    return torch.cat((T[..., :3, 3], quat), dim=-1)


def matrix_to_pose9d(T):
    if T.shape[-2:] != (4, 4):
        raise ValueError(f"T expected shape [...,4,4], got {tuple(T.shape)}")
    rot6d = matrix_to_rot6d(T[..., :3, :3])
    return torch.cat((T[..., :3, 3], rot6d), dim=-1)


def pose7d_to_pose9d(pose7d):
    return matrix_to_pose9d(pose7d_to_matrix(pose7d))


def pose9d_to_pose7d(pose9d):
    return matrix_to_pose7d(pose9d_to_matrix(pose9d))


def transform_point_cloud(points, T):
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points expected shape [B,N,3], got {tuple(points.shape)}")
    if T.ndim != 3 or T.shape[-2:] != (4, 4):
        raise ValueError(f"T expected shape [B,4,4], got {tuple(T.shape)}")
    if points.shape[0] != T.shape[0]:
        raise ValueError(f"batch mismatch: points {points.shape[0]}, T {T.shape[0]}")
    R = T[:, :3, :3]
    t = T[:, :3, 3]
    return torch.bmm(points, R.transpose(1, 2)) + t[:, None, :]


def rotation_geodesic_loss(R_pred, R_gt):
    if R_pred.shape[-2:] != (3, 3) or R_gt.shape[-2:] != (3, 3):
        raise ValueError(f"R_pred/R_gt expected [...,3,3], got {tuple(R_pred.shape)} and {tuple(R_gt.shape)}")
    R_rel = R_pred.transpose(-1, -2) @ R_gt
    trace = R_rel[..., 0, 0] + R_rel[..., 1, 1] + R_rel[..., 2, 2]
    cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cos).mean()
