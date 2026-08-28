"""Lightweight, headless visualizations for partial-cloud grasp hypotheses."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def project_camera(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    p = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    z = np.maximum(p[:, 2], 1e-6)
    return np.stack((p[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2], p[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2]), -1)


def save_partial_grasp_overlay(
    rgb: np.ndarray,
    intrinsic: np.ndarray,
    heatmap: np.ndarray,
    cup_mask: np.ndarray,
    first_contact_camera: np.ndarray,
    second_contact_camera: np.ndarray,
    tcp_camera: np.ndarray,
    output: str | Path,
    *,
    title: str = "single-view contact-pair grasp",
) -> None:
    """Save an RGB overlay; virtual contacts are deliberately shown in blue."""
    canvas = cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR).copy()
    mask = np.asarray(cup_mask).astype(bool)
    h = np.clip(np.asarray(heatmap, dtype=np.float32), 0.0, 1.0)
    color = cv2.applyColorMap((h * 255).astype(np.uint8), cv2.COLORMAP_JET)
    active = mask & (h > 0.03)
    canvas[active] = (0.42 * canvas[active] + 0.58 * color[active]).astype(np.uint8)
    p = project_camera(np.stack((first_contact_camera, second_contact_camera, tcp_camera[:3, 3])), intrinsic).astype(int)
    q0, q1, qt = [tuple(x) for x in p]
    cv2.line(canvas, q0, q1, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.drawMarker(canvas, q0, (0, 0, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.drawMarker(canvas, q1, (255, 80, 0), cv2.MARKER_TILTED_CROSS, 22, 2)
    cv2.drawMarker(canvas, qt, (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
    for axis, col in zip(np.eye(3, dtype=np.float32), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
        end = tcp_camera[:3, 3] + tcp_camera[:3, :3] @ axis * 0.06
        qe = tuple(project_camera(end[None], intrinsic)[0].astype(int))
        cv2.arrowedLine(canvas, qt, qe, col, 2, cv2.LINE_AA, tipLength=0.2)
    cv2.putText(canvas, title, (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "red=visible heat contact, blue=virtual opposite contact", (16, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), canvas)


def save_contact_pair_ply(points_camera: np.ndarray, heat: np.ndarray, first_contact: np.ndarray, second_contact: np.ndarray, tcp_camera: np.ndarray, output: str | Path) -> None:
    """Write a dependency-free colored PLY for later Open3D inspection."""
    p = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    h = np.clip(np.asarray(heat, dtype=np.float32).reshape(-1), 0.0, 1.0)
    if len(p) != len(h):
        raise ValueError("points_camera and heat must have equal length")
    # Jet-like RGB without requiring matplotlib; the exact colormap is not
    # important here, while preserving heat in the vertex colors is.
    rgb = np.stack((np.clip(1.5 * h, 0, 1), np.clip(1.5 - np.abs(2 * h - 1.0) * 2, 0, 1), np.clip(1.5 * (1 - h), 0, 1)), -1)
    vertices = [(float(x), float(y), float(z), int(255 * r), int(255 * g), int(255 * b)) for (x, y, z), (r, g, b) in zip(p, rgb)]
    extra = [(first_contact, (255, 0, 0)), (second_contact, (0, 80, 255)), (tcp_camera[:3, 3], (255, 255, 255))]
    extra_idx = []
    for point, color in extra:
        extra_idx.append(len(vertices)); vertices.append((float(point[0]), float(point[1]), float(point[2]), *color))
    edges = [(extra_idx[0], extra_idx[1]), (extra_idx[0], extra_idx[2]), (extra_idx[1], extra_idx[2])]
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write(f"element edge {len(edges)}\nproperty int vertex1\nproperty int vertex2\nend_header\n")
        for row in vertices:
            f.write("%.7f %.7f %.7f %d %d %d\n" % row)
        for e in edges:
            f.write(f"{e[0]} {e[1]}\n")
