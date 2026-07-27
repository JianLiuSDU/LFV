#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import zarr
from scipy.spatial import cKDTree


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.pipeline.tracking import load_episode_camera_params
from lfv.utils.config import load_config
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


DEFAULT_EPISODE = pathlib.Path("/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0")
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "hand_pouring.yaml"
DEFAULT_OUT_DIR = DEFAULT_EPISODE / "hamer_grasp_pseudo_label"


@dataclass
class Candidate:
    frame: int
    hand_id: str
    thumb_uv: np.ndarray
    index_uv: np.ndarray
    thumb_cam: np.ndarray
    index_cam: np.ndarray
    q_thumb_cam: np.ndarray
    q_index_cam: np.ndarray
    q_thumb_object: np.ndarray
    q_index_object: np.ndarray
    T_cam: np.ndarray
    T_object: np.ndarray
    width_m: float
    score: float
    finger_surface_dist_m: float
    heat_score: float


def normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return (vec / norm).astype(np.float32)


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float32).reshape(-1, 3)
    z = np.maximum(points_cam[:, 2], 1e-6)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1).astype(np.float32)


def unproject_uv_depth(uv: np.ndarray, z: float, K: np.ndarray) -> np.ndarray:
    u, v = float(uv[0]), float(uv[1])
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    return np.asarray([x, y, z], dtype=np.float32)


def heat_at_uv(heatmap: np.ndarray, uv: np.ndarray, radius: int = 3) -> float:
    h, w = heatmap.shape
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    x0, x1 = max(0, x - radius), min(w, x + radius + 1)
    y0, y1 = max(0, y - radius), min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return 0.0
    return float(np.max(heatmap[y0:y1, x0:x1]))


def valid_depth_at_uv(depth_m: np.ndarray, uv: np.ndarray, *, radius: int, fallback_radius: int) -> float | None:
    h, w = depth_m.shape
    x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
    for r in [radius, fallback_radius]:
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        vals = depth_m[y0:y1, x0:x1]
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if len(vals):
            return float(np.median(vals))
    return None


def matrix_to_grasp_row(T: np.ndarray, *, score: float, width: float, depth: float) -> np.ndarray:
    row = np.zeros(17, dtype=np.float32)
    row[0] = float(score)
    row[1] = float(width)
    row[2] = 0.02
    row[3] = float(depth)
    row[4:13] = T[:3, :3].reshape(-1)
    row[13:16] = T[:3, 3]
    return row


def grasp_points_from_T(T: np.ndarray, *, width: float, depth: float) -> np.ndarray:
    depth_base = 0.02
    local = np.asarray(
        [
            [-depth_base, -width / 2, 0.0],
            [depth, -width / 2, 0.0],
            [-depth_base, width / 2, 0.0],
            [depth, width / 2, 0.0],
            [0.0, 0.0, 0.0],
            [depth, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    return (T[:3, :3] @ local.T).T + T[:3, 3][None]


def draw_grasp_2d(canvas: np.ndarray, T_cam: np.ndarray, width: float, K: np.ndarray, *, label: str, color=(0, 0, 255)) -> None:
    pts = grasp_points_from_T(T_cam, width=width, depth=0.045)
    if np.any(pts[:, 2] <= 0):
        return
    uv = project(pts, K).round().astype(int)
    h, w = canvas.shape[:2]
    if not np.any((0 <= uv[:, 0]) & (uv[:, 0] < w) & (0 <= uv[:, 1]) & (uv[:, 1] < h)):
        return
    lb, lt, rb, rt, center, approach = [tuple(p) for p in uv]
    cv2.line(canvas, lb, lt, color, 3, cv2.LINE_AA)
    cv2.line(canvas, rb, rt, color, 3, cv2.LINE_AA)
    cv2.line(canvas, lb, rb, color, 3, cv2.LINE_AA)
    cv2.arrowedLine(canvas, center, approach, (255, 120, 30), 2, cv2.LINE_AA, tipLength=0.25)
    cv2.circle(canvas, center, 5, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (center[0] + 6, center[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def choose_frames(timing: dict, window_size: int) -> list[int]:
    frames = [int(f) for f in timing.get("contact_frames", [])]
    if frames:
        return frames[:window_size]
    start = timing.get("contact_start")
    if start is None:
        raise ValueError("contact_timing has no contact_start/contact_frames.")
    return [int(start) + i for i in range(window_size)]


def export_hamer_input(ep: pathlib.Path, frames: list[int], out_dir: pathlib.Path) -> None:
    rgb = zarr.open(str(ep / "rgb"), mode="r")
    out_dir.mkdir(parents=True, exist_ok=True)
    for frame in frames:
        img = np.asarray(rgb[frame], dtype=np.uint8)
        cv2.imwrite(str(out_dir / f"frame_{frame:06d}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def run_hamer(input_dir: pathlib.Path, output_dir: pathlib.Path, *, batch_size: int, overwrite: bool) -> None:
    keypoint_dir = output_dir / "skeleton2d"
    if keypoint_dir.exists() and list(keypoint_dir.glob("*_hand_2d.npy")) and not overwrite:
        print(f"[hamer_grasp] reuse existing HaMeR output: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(ROOT / "scripts" / "run_hamer_demo_env.sh"),
        "--img_folder",
        str(input_dir),
        "--out_folder",
        str(output_dir),
        "--batch_size",
        str(batch_size),
        "--save_skeleton2d",
        "--full_frame",
        "--file_type",
        "*.png",
    ]
    print("[hamer_grasp] run:", " ".join(cmd))
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def load_hand_keypoint_candidates(hamer_out: pathlib.Path, frame: int) -> list[tuple[str, np.ndarray]]:
    paths = sorted((hamer_out / "skeleton2d").glob(f"frame_{frame:06d}_person*_hand_2d.npy"))
    candidates = []
    for path in paths:
        arr = np.load(path).astype(np.float32)
        if arr.shape == (21, 3):
            hand_id = path.stem.replace(f"frame_{frame:06d}_", "")
            candidates.append((hand_id, arr))
    return candidates


def select_best_hand(kps: list[tuple[str, np.ndarray]], hand_mask: np.ndarray, heatmap: np.ndarray, *, min_conf: float) -> tuple[str, np.ndarray] | None:
    if not kps:
        return None
    h, w = hand_mask.shape
    dilated_hand = cv2.dilate(hand_mask.astype(np.uint8), np.ones((9, 9), dtype=np.uint8), iterations=1).astype(bool)
    best = None
    for hand_id, kp in kps:
        thumb = kp[4]
        index = kp[8]
        if float(min(thumb[2], index[2])) < min_conf:
            continue
        score = 0.5 * (float(thumb[2]) + float(index[2]))
        for uv in [thumb[:2], index[:2]]:
            x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 0 <= x < w and 0 <= y < h:
                score += 0.2 if dilated_hand[y, x] else 0.0
                score += 0.4 * heat_at_uv(heatmap, uv, radius=4)
        if best is None or score > best[0]:
            best = (score, hand_id, kp)
    if best is None:
        return None
    return best[1], best[2]


def build_candidate(
    *,
    frame: int,
    hand_id: str,
    kp: np.ndarray,
    depth_m: np.ndarray,
    K: np.ndarray,
    object_points: np.ndarray,
    object_normals: np.ndarray,
    object_center: np.ndarray,
    object_tree: cKDTree,
    heatmap: np.ndarray,
    point_heat: np.ndarray,
    depth_radius: int,
    depth_fallback_radius: int,
    width_margin: float,
    tcp_to_contact_offset: float,
    min_width: float,
    max_width: float,
    max_finger_surface_dist: float,
) -> Candidate | None:
    thumb_uv = kp[4, :2].astype(np.float32)
    index_uv = kp[8, :2].astype(np.float32)
    z_thumb = valid_depth_at_uv(depth_m, thumb_uv, radius=depth_radius, fallback_radius=depth_fallback_radius)
    z_index = valid_depth_at_uv(depth_m, index_uv, radius=depth_radius, fallback_radius=depth_fallback_radius)
    if z_thumb is None or z_index is None:
        return None
    thumb_cam = unproject_uv_depth(thumb_uv, z_thumb, K)
    index_cam = unproject_uv_depth(index_uv, z_index, K)

    d_thumb, idx_thumb = object_tree.query(thumb_cam, k=1)
    d_index, idx_index = object_tree.query(index_cam, k=1)
    finger_surface_dist = float(max(d_thumb, d_index))
    if finger_surface_dist > max_finger_surface_dist:
        return None

    q_thumb = object_points[int(idx_thumb)].astype(np.float32)
    q_index = object_points[int(idx_index)].astype(np.float32)
    contact_vec = q_index - q_thumb
    contact_dist = float(np.linalg.norm(contact_vec))
    width = contact_dist + float(width_margin)
    if not (min_width <= width <= max_width) or contact_dist < 1e-4:
        return None

    closing = normalize(contact_vec)
    normal = object_normals[int(idx_thumb)] + object_normals[int(idx_index)]
    if float(np.linalg.norm(normal)) < 1e-6:
        normal = object_points[int(idx_thumb)] - object_center
    try:
        approach = normalize(-normal)
    except ValueError:
        approach = normalize(0.5 * (q_thumb + q_index) - np.zeros(3, dtype=np.float32))
    approach = approach - closing * float(np.dot(approach, closing))
    if float(np.linalg.norm(approach)) < 1e-5:
        approach = normalize(0.5 * (q_thumb + q_index))
        approach = approach - closing * float(np.dot(approach, closing))
    approach = normalize(approach)
    binormal = normalize(np.cross(approach, closing))
    closing = normalize(np.cross(binormal, approach))

    center = 0.5 * (q_thumb + q_index)
    tcp = center - approach * float(tcp_to_contact_offset)
    R = np.stack([approach, closing, binormal], axis=1).astype(np.float32)
    T_cam = np.eye(4, dtype=np.float32)
    T_cam[:3, :3] = R
    T_cam[:3, 3] = tcp.astype(np.float32)
    T_object = T_cam.copy()
    T_object[:3, 3] = (tcp - object_center).astype(np.float32)
    heat_score = max(heat_at_uv(heatmap, thumb_uv), heat_at_uv(heatmap, index_uv), float(point_heat[int(idx_thumb)]), float(point_heat[int(idx_index)]))
    score = float(heat_score) + float(0.5 * (kp[4, 2] + kp[8, 2])) - finger_surface_dist / max(max_finger_surface_dist, 1e-6)
    return Candidate(
        frame=frame,
        hand_id=hand_id,
        thumb_uv=thumb_uv,
        index_uv=index_uv,
        thumb_cam=thumb_cam,
        index_cam=index_cam,
        q_thumb_cam=q_thumb,
        q_index_cam=q_index,
        q_thumb_object=(q_thumb - object_center).astype(np.float32),
        q_index_object=(q_index - object_center).astype(np.float32),
        T_cam=T_cam,
        T_object=T_object,
        width_m=float(width),
        score=score,
        finger_surface_dist_m=finger_surface_dist,
        heat_score=float(heat_score),
    )


def rot_distance(Ra: np.ndarray, Rb: np.ndarray) -> float:
    val = (float(np.trace(Ra.T @ Rb)) - 1.0) * 0.5
    return float(np.arccos(np.clip(val, -1.0, 1.0)))


def select_representative(candidates: list[Candidate]) -> tuple[int, dict]:
    if not candidates:
        raise ValueError("No valid thumb-index grasp candidates.")
    if len(candidates) == 1:
        return 0, {"mean_pairwise_translation_m": 0.0, "mean_pairwise_rotation_rad": 0.0, "mean_pairwise_width_m": 0.0}
    scores = []
    trans_d = []
    rot_d = []
    width_d = []
    for i, ci in enumerate(candidates):
        dist = 0.0
        for j, cj in enumerate(candidates):
            if i == j:
                continue
            dt = float(np.linalg.norm(ci.T_object[:3, 3] - cj.T_object[:3, 3]))
            dr = rot_distance(ci.T_object[:3, :3], cj.T_object[:3, :3])
            dw = abs(ci.width_m - cj.width_m)
            dist += dt / 0.03 + dr / 0.5 + dw / 0.02
            trans_d.append(dt)
            rot_d.append(dr)
            width_d.append(dw)
        scores.append(dist)
    return int(np.argmin(scores)), {
        "mean_pairwise_translation_m": float(np.mean(trans_d)) if trans_d else 0.0,
        "mean_pairwise_rotation_rad": float(np.mean(rot_d)) if rot_d else 0.0,
        "mean_pairwise_width_m": float(np.mean(width_d)) if width_d else 0.0,
    }


def save_overlay(
    rgb: np.ndarray,
    heatmap: np.ndarray,
    object_mask: np.ndarray,
    candidate: Candidate,
    K: np.ndarray,
    out_path: pathlib.Path,
) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    heat_color = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    alpha = np.clip(heatmap[..., None], 0, 1) * 0.55
    bgr = (bgr * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    contours, _ = cv2.findContours(object_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(bgr, contours, -1, (0, 255, 255), 1)
    draw_grasp_2d(bgr, candidate.T_cam, candidate.width_m, K, label=f"frame {candidate.frame}")
    for uv, color, text in [(candidate.thumb_uv, (255, 0, 255), "thumb"), (candidate.index_uv, (0, 255, 0), "index")]:
        x, y = int(round(float(uv[0]))), int(round(float(uv[1])))
        cv2.circle(bgr, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.putText(bgr, text, (x + 5, y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    for pt, color in [(candidate.q_thumb_cam, (255, 0, 255)), (candidate.q_index_cam, (0, 255, 0))]:
        uv = project(pt[None], K)[0].round().astype(int)
        cv2.drawMarker(bgr, tuple(uv), color, markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2)
    cv2.imwrite(str(out_path), bgr)


def save_window_candidates(rgb_video, candidates: list[Candidate], K: np.ndarray, out_path: pathlib.Path) -> None:
    if not candidates:
        return
    cols = min(4, len(candidates))
    rows = int(np.ceil(len(candidates) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, cand in zip(axes, candidates):
        canvas = cv2.cvtColor(np.asarray(rgb_video[cand.frame], dtype=np.uint8), cv2.COLOR_RGB2BGR)
        draw_grasp_2d(canvas, cand.T_cam, cand.width_m, K, label=f"{cand.frame}", color=(0, 0, 255))
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        ax.imshow(canvas)
        ax.set_title(f"f{cand.frame} w={cand.width_m:.3f} d={cand.finger_surface_dist_m:.3f}")
        ax.axis("off")
    for ax in axes[len(candidates):]:
        ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=140)
    plt.close(fig)


def save_3d(points: np.ndarray, heat: np.ndarray, candidate: Candidate, out_path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heat, s=5, cmap="magma", vmin=0, vmax=1, alpha=0.85)
    gripper = grasp_points_from_T(candidate.T_cam, width=candidate.width_m, depth=0.045)
    for a, b, color in [(0, 1, "tab:red"), (2, 3, "tab:red"), (0, 2, "tab:red"), (4, 5, "tab:blue")]:
        ax.plot([gripper[a, 0], gripper[b, 0]], [gripper[a, 1], gripper[b, 1]], [gripper[a, 2], gripper[b, 2]], color=color, linewidth=3)
    ax.scatter(candidate.q_thumb_cam[0], candidate.q_thumb_cam[1], candidate.q_thumb_cam[2], c="magenta", s=70, label="thumb surface")
    ax.scatter(candidate.q_index_cam[0], candidate.q_index_cam[1], candidate.q_index_cam[2], c="lime", s=70, label="index surface")
    ax.set_xlabel("x camera (m)")
    ax.set_ylabel("y camera (m)")
    ax.set_zlabel("z camera (m)")
    ax.view_init(elev=20, azim=-65)
    ax.legend(loc="upper right")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate one episode thumb-index grasp pseudo label using HaMeR keypoints.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--episode-dir", default=str(DEFAULT_EPISODE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-hamer", action="store_true", help="Do not run HaMeR; only consume existing HaMeR outputs.")
    parser.add_argument("--min-keypoint-conf", type=float, default=0.2)
    parser.add_argument("--depth-radius", type=int, default=4)
    parser.add_argument("--depth-fallback-radius", type=int, default=14)
    parser.add_argument("--width-margin", type=float, default=0.012)
    parser.add_argument("--tcp-to-contact-offset", type=float, default=0.045)
    parser.add_argument("--min-width", type=float, default=0.015)
    parser.add_argument("--max-width", type=float, default=0.085)
    parser.add_argument("--max-finger-surface-dist", type=float, default=0.070)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ep = pathlib.Path(args.episode_dir)
    out_dir = pathlib.Path(args.out_dir)
    input_dir = out_dir / "hamer_input"
    hamer_out = out_dir / "hamer_output"
    viz_dir = out_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)

    with (ep / "contact_timing" / "contact_timing.json").open("r", encoding="utf-8") as f:
        timing = json.load(f)
    frames = choose_frames(timing, args.window_size)
    export_hamer_input(ep, frames, input_dir)
    if not args.skip_hamer:
        run_hamer(input_dir, hamer_out, batch_size=args.batch_size, overwrite=args.overwrite)

    K, depth_scale, _meta = load_episode_camera_params(ep, cfg)
    rgb_video = zarr.open(str(ep / "rgb"), mode="r")
    depth_raw = zarr.open(str(ep / "depth"), mode="r")
    heat_data = np.load(ep / "contact_heatmap" / "contact_heatmap.npz")
    points = heat_data["points_camera"].astype(np.float32)
    normals = heat_data["normals_camera"].astype(np.float32)
    object_center = heat_data["object_center_camera"].astype(np.float32)
    point_heat = heat_data["contact_heat"].astype(np.float32)
    heatmap = heat_data["heatmap_2d"].astype(np.float32)
    object_mask = heat_data["anchor_object_mask"].astype(bool)
    object_tree = cKDTree(points)

    candidates: list[Candidate] = []
    per_frame_meta = []
    for frame in frames:
        hand_mask_path = ep / "hand_mask" / f"frame_{frame:06d}.npy"
        if not hand_mask_path.exists():
            per_frame_meta.append({"frame": int(frame), "status": "missing_hand_mask"})
            continue
        hand_mask = np.load(hand_mask_path, allow_pickle=True).astype(bool)
        kp_candidates = load_hand_keypoint_candidates(hamer_out, frame)
        selected = select_best_hand(kp_candidates, hand_mask, heatmap, min_conf=args.min_keypoint_conf)
        if selected is None:
            per_frame_meta.append({"frame": int(frame), "status": "missing_or_low_conf_keypoints", "num_hands": len(kp_candidates)})
            continue
        hand_id, kp = selected
        depth_m = np.asarray(depth_raw[frame], dtype=np.float32) * float(depth_scale)
        cand = build_candidate(
            frame=frame,
            hand_id=hand_id,
            kp=kp,
            depth_m=depth_m,
            K=K,
            object_points=points,
            object_normals=normals,
            object_center=object_center,
            object_tree=object_tree,
            heatmap=heatmap,
            point_heat=point_heat,
            depth_radius=args.depth_radius,
            depth_fallback_radius=args.depth_fallback_radius,
            width_margin=args.width_margin,
            tcp_to_contact_offset=args.tcp_to_contact_offset,
            min_width=args.min_width,
            max_width=args.max_width,
            max_finger_surface_dist=args.max_finger_surface_dist,
        )
        if cand is None:
            per_frame_meta.append({"frame": int(frame), "status": "candidate_filtered", "hand_id": hand_id})
            continue
        candidates.append(cand)
        per_frame_meta.append(
            {
                "frame": int(frame),
                "status": "ok",
                "hand_id": hand_id,
                "width_m": cand.width_m,
                "finger_surface_dist_m": cand.finger_surface_dist_m,
                "heat_score": cand.heat_score,
                "score": cand.score,
                "thumb_uv": cand.thumb_uv.astype(float).tolist(),
                "index_uv": cand.index_uv.astype(float).tolist(),
            }
        )

    selected_idx, consistency = select_representative(candidates)
    selected = candidates[selected_idx]
    valid_ratio = len(candidates) / max(len(frames), 1)
    mean_surface_dist = float(np.mean([c.finger_surface_dist_m for c in candidates])) if candidates else float("inf")
    confidence = float(
        np.clip(
            valid_ratio
            * np.exp(-mean_surface_dist / max(args.max_finger_surface_dist, 1e-6))
            * np.exp(-consistency["mean_pairwise_translation_m"] / 0.04)
            * max(selected.heat_score, 0.05),
            0.0,
            1.0,
        )
    )
    quality = "good"
    if len(candidates) < max(2, len(frames) // 2) or confidence < 0.15:
        quality = "review"
    if len(candidates) == 0:
        quality = "reject"

    candidate_T_cam = np.stack([c.T_cam for c in candidates], axis=0).astype(np.float32)
    candidate_T_object = np.stack([c.T_object for c in candidates], axis=0).astype(np.float32)
    candidate_width = np.asarray([c.width_m for c in candidates], dtype=np.float32)
    candidate_frames = np.asarray([c.frame for c in candidates], dtype=np.int32)
    rotation_6d = selected.T_object[:3, :2].reshape(-1).astype(np.float32)

    np.savez_compressed(
        out_dir / "grasp_pseudo_label.npz",
        T_grasp_cam=selected.T_cam.astype(np.float32),
        T_grasp_object=selected.T_object.astype(np.float32),
        rotation_6d=rotation_6d,
        translation_object=selected.T_object[:3, 3].astype(np.float32),
        width_m=np.asarray(selected.width_m, dtype=np.float32),
        q_thumb_object=selected.q_thumb_object.astype(np.float32),
        q_index_object=selected.q_index_object.astype(np.float32),
        q_thumb_cam=selected.q_thumb_cam.astype(np.float32),
        q_index_cam=selected.q_index_cam.astype(np.float32),
        candidate_T_cam=candidate_T_cam,
        candidate_T_object=candidate_T_object,
        candidate_width_m=candidate_width,
        candidate_frames=candidate_frames,
        valid=np.asarray(quality != "reject"),
        confidence=np.asarray(confidence, dtype=np.float32),
        selected_frame=np.asarray(selected.frame, dtype=np.int32),
    )

    save_overlay(np.asarray(rgb_video[selected.frame]), heatmap, object_mask, selected, K, viz_dir / "selected_grasp_overlay_2d.png")
    save_window_candidates(rgb_video, candidates, K, viz_dir / "window_candidates_2d.png")
    save_3d(points, point_heat, selected, viz_dir / "selected_grasp_3d.png")

    row = matrix_to_grasp_row(selected.T_cam, score=confidence, width=selected.width_m, depth=0.045)
    np.save(out_dir / "selected_grasp_graspnet_row.npy", row)
    meta = {
        "episode_dir": str(ep),
        "output_dir": str(out_dir),
        "frames_requested": [int(f) for f in frames],
        "frames_valid": [int(c.frame) for c in candidates],
        "selected_frame": int(selected.frame),
        "selected_hand_id": selected.hand_id,
        "quality": quality,
        "confidence": confidence,
        "valid_candidate_count": int(len(candidates)),
        "window_size": int(len(frames)),
        "approach_convention": "T columns: X=approach, Y=closing, Z=binormal; object coordinates are camera axes translated by object_center_camera.",
        "width_m": float(selected.width_m),
        "translation_object": selected.T_object[:3, 3].astype(float).tolist(),
        "q_thumb_object": selected.q_thumb_object.astype(float).tolist(),
        "q_index_object": selected.q_index_object.astype(float).tolist(),
        "mean_finger_surface_dist_m": mean_surface_dist,
        "consistency": consistency,
        "per_frame": per_frame_meta,
        "outputs": {
            "npz": str(out_dir / "grasp_pseudo_label.npz"),
            "selected_grasp_graspnet_row": str(out_dir / "selected_grasp_graspnet_row.npy"),
            "selected_grasp_overlay_2d": str(viz_dir / "selected_grasp_overlay_2d.png"),
            "window_candidates_2d": str(viz_dir / "window_candidates_2d.png"),
            "selected_grasp_3d": str(viz_dir / "selected_grasp_3d.png"),
            "hamer_input": str(input_dir),
            "hamer_output": str(hamer_out),
        },
    }
    with (out_dir / "grasp_pseudo_label_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
