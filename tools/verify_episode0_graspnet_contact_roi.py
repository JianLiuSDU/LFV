#!/usr/bin/env python
from __future__ import annotations

import argparse
import io
import json
import pathlib
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import requests
import zarr


ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lfv.pipeline.tracking import load_episode_camera_params, project_to_2d
from lfv.utils.config import load_config
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


DEFAULT_EPISODE = pathlib.Path("/media/ljian/lj/data_3d/hand_pouring_lfv/episode_0")
DEFAULT_OUT_DIR = DEFAULT_EPISODE / "graspnet_contact_roi_verify"
GRASPNET_API_URL = "http://127.0.0.1:5000/predict_grasp"


def _normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        raise ValueError(f"Cannot normalize near-zero vector: {vec}")
    return (vec / norm).astype(np.float32)


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask > 0.5, dtype=bool)


def project(points_cam: np.ndarray, K: np.ndarray) -> np.ndarray:
    points_cam = np.asarray(points_cam, dtype=np.float32)
    z = np.maximum(points_cam[:, 2], 1e-6)
    u = K[0, 0] * points_cam[:, 0] / z + K[0, 2]
    v = K[1, 1] * points_cam[:, 1] / z + K[1, 2]
    return np.stack([u, v], axis=1)


def grasp_row_to_matrix_cam(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    if row.shape[0] < 16:
        raise ValueError(f"Grasp row should have at least 16 values, got {row.shape}")
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = row[4:13].reshape(3, 3)
    T[:3, 3] = row[13:16]
    return T


def matrix_cam_to_grasp_row(T: np.ndarray, *, score: float = 1.0, width: float = 0.035, depth: float = 0.045) -> np.ndarray:
    row = np.zeros(17, dtype=np.float32)
    row[0] = float(score)
    row[1] = float(width)
    row[2] = 0.02
    row[3] = float(depth)
    row[4:13] = T[:3, :3].reshape(-1)
    row[13:16] = T[:3, 3]
    return row


def grasp_points(row: np.ndarray) -> np.ndarray:
    row = np.asarray(row, dtype=np.float32).reshape(-1)
    width = float(max(row[1], 0.015))
    depth = float(max(row[3], 0.025))
    R = row[4:13].reshape(3, 3)
    center = row[13:16]
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
    return (R @ local.T).T + center[None]


def draw_grasp_2d(canvas: np.ndarray, row: np.ndarray, K: np.ndarray, color: tuple[int, int, int], thickness: int, label: str) -> None:
    pts = grasp_points(row)
    if np.any(pts[:, 2] <= 0):
        return
    uv = project(pts, K).round().astype(int)
    h, w = canvas.shape[:2]
    if not np.any((0 <= uv[:, 0]) & (uv[:, 0] < w) & (0 <= uv[:, 1]) & (uv[:, 1] < h)):
        return
    lb, lt, rb, rt, center, approach = [tuple(p) for p in uv]
    cv2.line(canvas, lb, lt, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, rb, rt, color, thickness, cv2.LINE_AA)
    cv2.line(canvas, lb, rb, color, thickness, cv2.LINE_AA)
    cv2.arrowedLine(canvas, center, approach, (255, 120, 30), max(1, thickness), cv2.LINE_AA, tipLength=0.25)
    cv2.circle(canvas, center, 3 + thickness, color, -1, cv2.LINE_AA)
    cv2.putText(canvas, label, (center[0] + 5, center[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


def call_graspnet_api(rgb: np.ndarray, depth_m: np.ndarray, mask: np.ndarray, K: np.ndarray, url: str) -> np.ndarray:
    mem_file = io.BytesIO()
    np.savez_compressed(
        mem_file,
        rgb=np.asarray(rgb, dtype=np.uint8),
        depth=np.asarray(depth_m, dtype=np.float32),
        mask=np.asarray(mask, dtype=np.uint8),
        K=np.asarray(K, dtype=np.float32),
    )
    mem_file.seek(0)
    response = requests.post(url, files={"file": ("data.npz", mem_file)}, timeout=240)
    response.raise_for_status()
    result = response.json()
    grasps = result.get("grasps", [])
    if not grasps:
        raise RuntimeError(f"GraspNet returned no grasps: {result.get('message')}")
    return np.asarray(grasps, dtype=np.float32)


def heat_at_uv(heatmap: np.ndarray, uv: np.ndarray, radius: int = 2) -> float:
    h, w = heatmap.shape
    uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
    vals = []
    for x, y in uv:
        cx, cy = int(round(float(x))), int(round(float(y)))
        x0, x1 = max(0, cx - radius), min(w, cx + radius + 1)
        y0, y1 = max(0, cy - radius), min(h, cy + radius + 1)
        if x0 < x1 and y0 < y1:
            vals.append(float(np.max(heatmap[y0:y1, x0:x1])))
    return max(vals) if vals else 0.0


def score_candidates(candidates: np.ndarray, heatmap: np.ndarray, K: np.ndarray, top_n: int) -> tuple[np.ndarray, list[dict]]:
    candidates = np.asarray(candidates, dtype=np.float32)
    if len(candidates) == 0:
        return candidates, []
    max_score = max(float(np.max(candidates[:top_n, 0])), 1e-8)
    records = []
    for idx, row in enumerate(candidates[:top_n]):
        pts = grasp_points(row)
        uv = project(pts[[1, 3, 4, 5]], K)
        center_uv = project(row[13:16][None], K)[0]
        center_heat = heat_at_uv(heatmap, center_uv[None], radius=3)
        endpoint_heat = heat_at_uv(heatmap, uv, radius=3)
        contact_score = max(center_heat, endpoint_heat)
        score_norm = float(row[0]) / max_score
        final_score = 0.5 * score_norm + 0.5 * contact_score
        records.append(
            {
                "idx": int(idx),
                "graspnet_score": float(row[0]),
                "score_norm": score_norm,
                "center_heat": center_heat,
                "endpoint_heat": endpoint_heat,
                "contact_score": contact_score,
                "final_score": final_score,
                "center_uv": center_uv.astype(float).tolist(),
                "center_camera": row[13:16].astype(float).tolist(),
            }
        )
    order = sorted(records, key=lambda item: item["final_score"], reverse=True)
    filtered = np.asarray([candidates[item["idx"]] for item in order], dtype=np.float32)
    return filtered, order


def fallback_contact_grasp(points: np.ndarray, normals: np.ndarray, heat: np.ndarray) -> tuple[np.ndarray, dict]:
    heat = np.asarray(heat, dtype=np.float32)
    high = heat >= max(0.4, float(np.max(heat)) * 0.75)
    if int(np.sum(high)) < 4:
        high = heat >= max(0.1, float(np.max(heat)) * 0.5)
    if int(np.sum(high)) < 4:
        high[np.argmax(heat)] = True
    weights = np.maximum(heat[high], 1e-4)
    pts = points[high].astype(np.float32)
    center = np.sum(pts * weights[:, None], axis=0) / max(float(np.sum(weights)), 1e-8)
    normal = np.sum(normals[high].astype(np.float32) * weights[:, None], axis=0)
    approach = _normalize(-normal)
    if approach[2] < 0:
        approach = -approach
    centered = pts - np.mean(pts, axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    closing = vh[0].astype(np.float32)
    closing = closing - approach * float(closing @ approach)
    if float(np.linalg.norm(closing)) < 1e-6:
        closing = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        closing = closing - approach * float(closing @ approach)
    closing = _normalize(closing)
    height = _normalize(np.cross(approach, closing))
    closing = _normalize(np.cross(height, approach))
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.stack([approach, closing, height], axis=1)
    T[:3, 3] = center
    row = matrix_cam_to_grasp_row(T, score=float(np.max(heat)), width=0.035, depth=0.045)
    meta = {
        "source": "fallback_contact_heat",
        "high_heat_points": int(np.sum(high)),
        "center_camera": center.astype(float).tolist(),
        "approach_camera": approach.astype(float).tolist(),
        "closing_camera": closing.astype(float).tolist(),
    }
    return row, meta


def save_heat_overlay(rgb: np.ndarray, heatmap: np.ndarray, out_path: pathlib.Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(rgb)
    ax.imshow(heatmap, cmap="magma", alpha=np.clip(heatmap, 0, 1) * 0.75, vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_mask_overlay(rgb: np.ndarray, mask: np.ndarray, out_path: pathlib.Path, title: str) -> None:
    canvas = rgb.copy()
    canvas[mask > 0] = (0.45 * canvas[mask > 0] + 0.55 * np.asarray([255, 210, 0])).astype(np.uint8)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(canvas)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def save_grasp_overlay(rgb: np.ndarray, heatmap: np.ndarray, K: np.ndarray, candidates: np.ndarray, selected: np.ndarray, out_path: pathlib.Path) -> None:
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    heat_color = cv2.applyColorMap((np.clip(heatmap, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_MAGMA)
    alpha = np.clip(heatmap[..., None], 0, 1) * 0.65
    bgr = (bgr * (1 - alpha) + heat_color * alpha).astype(np.uint8)
    top_k = min(20, len(candidates))
    for idx in range(top_k - 1, -1, -1):
        draw_grasp_2d(bgr, candidates[idx], K, (40, 220, 80), 1, f"{idx}:{candidates[idx, 0]:.2f}")
    draw_grasp_2d(bgr, selected, K, (0, 0, 255), 3, "selected")
    cv2.imwrite(str(out_path), bgr)


def save_grasp_3d(points: np.ndarray, heat: np.ndarray, selected: np.ndarray, out_path: pathlib.Path) -> None:
    fig = plt.figure(figsize=(9, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=heat, s=5, cmap="magma", vmin=0, vmax=1)
    pts = grasp_points(selected)
    segments = [(0, 1), (2, 3), (0, 2), (4, 5)]
    for a, b in segments:
        color = "tab:red" if (a, b) != (4, 5) else "tab:blue"
        ax.plot([pts[a, 0], pts[b, 0]], [pts[a, 1], pts[b, 1]], [pts[a, 2], pts[b, 2]], color=color, linewidth=3)
    ax.scatter(pts[4, 0], pts[4, 1], pts[4, 2], c="red", s=60)
    ax.set_xlabel("x camera (m)")
    ax.set_ylabel("y camera (m)")
    ax.set_zlabel("z camera (m)")
    ax.view_init(elev=20, azim=-65)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def show_open3d(points: np.ndarray, heat: np.ndarray, selected: np.ndarray) -> None:
    import open3d as o3d
    import matplotlib.cm as cm

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    colors = cm.get_cmap("magma")(np.clip(heat, 0, 1))[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    pts = grasp_points(selected).astype(np.float64)
    lines = np.asarray([[0, 1], [2, 3], [0, 2], [4, 5]], dtype=np.int32)
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(pts),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(
        np.asarray([[1, 0, 0], [1, 0, 0], [1, 0, 0], [0, 0.4, 1]], dtype=np.float64)
    )
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04, origin=pts[4])
    o3d.visualization.draw_geometries([pcd, line_set, frame], window_name="episode_0 contact ROI grasp")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify contact-ROI constrained GraspNet on hand_pouring episode_0.")
    parser.add_argument("--config", default="configs/pipeline/hand_pouring.yaml")
    parser.add_argument("--episode-dir", default=str(DEFAULT_EPISODE))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--api-url", default=GRASPNET_API_URL)
    parser.add_argument("--call-api", action="store_true", help="Call GraspNet API for full object and contact ROI masks.")
    parser.add_argument("--no-fallback", action="store_true", help="Fail if GraspNet API is unavailable or returns no usable candidates.")
    parser.add_argument("--roi-threshold", type=float, default=0.25)
    parser.add_argument("--roi-dilate", type=int, default=7)
    parser.add_argument("--top-n", type=int, default=80)
    parser.add_argument("--show-open3d", action="store_true")
    args = parser.parse_args()

    cfg_path = pathlib.Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    cfg = load_config(cfg_path)
    ep = pathlib.Path(args.episode_dir)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    heat_data = np.load(ep / "contact_heatmap" / "contact_heatmap.npz")
    with (ep / "contact_heatmap" / "contact_heatmap_meta.json").open("r", encoding="utf-8") as f:
        heat_meta = json.load(f)
    anchor_frame = int(heat_meta["anchor_frame"])
    rgb = np.asarray(zarr.open(str(ep / "rgb"), mode="r")[anchor_frame])
    depth_raw = np.asarray(zarr.open(str(ep / "depth"), mode="r")[anchor_frame])
    K, depth_scale, _meta = load_episode_camera_params(ep, cfg)
    depth_m = depth_raw.astype(np.float32) * float(depth_scale)
    object_mask = _as_bool_mask(np.load(ep / "sam_mask" / "affordance_mask.npy", allow_pickle=True))
    heatmap = heat_data["heatmap_2d"].astype(np.float32)
    points = heat_data["points_camera"].astype(np.float32)
    normals = heat_data["normals_camera"].astype(np.float32)
    point_heat = heat_data["contact_heat"].astype(np.float32)

    roi = (heatmap >= float(args.roi_threshold)) & object_mask
    if args.roi_dilate > 0:
        kernel = np.ones((int(args.roi_dilate), int(args.roi_dilate)), dtype=np.uint8)
        roi = cv2.dilate(roi.astype(np.uint8), kernel, iterations=1).astype(bool) & object_mask

    cv2.imwrite(str(out_dir / "graspnet_input_rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_dir / "full_object_mask.png"), object_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "contact_roi_mask.png"), roi.astype(np.uint8) * 255)
    cv2.imwrite(str(out_dir / "depth_anchor_u16_mm.png"), np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16))
    save_mask_overlay(rgb, object_mask, out_dir / "full_object_mask_overlay.png", "full cup object mask")
    save_mask_overlay(rgb, roi, out_dir / "contact_roi_mask_overlay.png", "contact ROI mask")
    save_heat_overlay(rgb, heatmap, out_dir / "contact_heat_overlay.png", "contact heatmap")

    api_error = None
    candidates_full = np.zeros((0, 17), dtype=np.float32)
    candidates_roi = np.zeros((0, 17), dtype=np.float32)
    if args.call_api:
        try:
            candidates_full = call_graspnet_api(rgb, depth_m, object_mask.astype(np.uint8), K, args.api_url)
            np.save(out_dir / "graspnet_candidates_full.npy", candidates_full)
        except Exception as exc:
            api_error = f"full_object_api: {repr(exc)}"
        try:
            candidates_roi = call_graspnet_api(rgb, depth_m, roi.astype(np.uint8), K, args.api_url)
            np.save(out_dir / "graspnet_candidates_roi.npy", candidates_roi)
        except Exception as exc:
            msg = f"roi_api: {repr(exc)}"
            api_error = msg if api_error is None else f"{api_error}; {msg}"

    filtered_full, full_records = score_candidates(candidates_full, heatmap, K, min(args.top_n, len(candidates_full))) if len(candidates_full) else (candidates_full, [])
    filtered_roi, roi_records = score_candidates(candidates_roi, heatmap, K, min(args.top_n, len(candidates_roi))) if len(candidates_roi) else (candidates_roi, [])
    candidate_pool = filtered_roi if len(filtered_roi) else filtered_full
    records = roi_records if len(filtered_roi) else full_records

    if len(candidate_pool):
        selected = candidate_pool[0]
        selected_source = "graspnet_roi_filtered" if len(filtered_roi) else "graspnet_full_contact_filtered"
        fallback_meta = None
    else:
        if args.no_fallback:
            raise RuntimeError(f"No GraspNet candidates available. api_error={api_error}")
        selected, fallback_meta = fallback_contact_grasp(points, normals, point_heat)
        selected_source = "fallback_contact_heat"
        candidate_pool = selected[None]
        records = []

    np.save(out_dir / "filtered_candidates.npy", candidate_pool.astype(np.float32))
    np.save(out_dir / "selected_grasp.npy", selected.astype(np.float32))
    np.save(out_dir / "selected_G_cam.npy", grasp_row_to_matrix_cam(selected).astype(np.float32))
    save_grasp_overlay(rgb, heatmap, K, candidate_pool, selected, out_dir / "selected_grasp_overlay.png")
    save_grasp_3d(points, point_heat, selected, out_dir / "selected_grasp_3d.png")

    report = {
        "episode_dir": str(ep),
        "anchor_frame": anchor_frame,
        "api_url": args.api_url,
        "call_api": bool(args.call_api),
        "api_error": api_error,
        "selected_source": selected_source,
        "selected_score": float(selected[0]),
        "selected_center_camera": selected[13:16].astype(float).tolist(),
        "full_candidates": int(len(candidates_full)),
        "roi_candidates": int(len(candidates_roi)),
        "filtered_candidates": int(len(candidate_pool)),
        "roi_threshold": float(args.roi_threshold),
        "roi_pixels": int(np.sum(roi)),
        "full_records_top": records[:20],
        "fallback_meta": fallback_meta,
        "outputs": {
            "selected_grasp_overlay": str(out_dir / "selected_grasp_overlay.png"),
            "selected_grasp_3d": str(out_dir / "selected_grasp_3d.png"),
            "contact_roi_mask": str(out_dir / "contact_roi_mask.png"),
            "contact_heat_overlay": str(out_dir / "contact_heat_overlay.png"),
        },
    }
    with (out_dir / "graspnet_contact_filter_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(json.dumps(report, indent=2, ensure_ascii=False))

    if args.show_open3d:
        show_open3d(points, point_heat, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
