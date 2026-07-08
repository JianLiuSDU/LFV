from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import zarr

from lfv.data.episode_io import iter_processed_episodes
from lfv.utils.imagecodecs import register_image_codecs


register_image_codecs()


def _add_tapip3d_to_path() -> Path:
    root = Path(__file__).resolve().parents[2] / "third_party" / "tapip3d"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


def _device(cfg) -> str:
    requested = str(cfg.runtime.device)
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return requested


def _intrinsics_dict_to_matrix(intrinsics_raw: dict) -> np.ndarray:
    fx = intrinsics_raw.get("fx", intrinsics_raw.get("f"))
    fy = intrinsics_raw.get("fy", intrinsics_raw.get("f"))
    cx = intrinsics_raw.get("cx", intrinsics_raw.get("ppx"))
    cy = intrinsics_raw.get("cy", intrinsics_raw.get("ppy"))
    if fx is None or fy is None or cx is None or cy is None:
        raise ValueError(f"Invalid intrinsics dictionary: {intrinsics_raw}")
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def load_episode_camera_params(ep_path: str | Path, cfg) -> tuple[np.ndarray, float, dict]:
    meta_path = Path(ep_path) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing episode meta.json: {meta_path}")

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)

    intrinsics_source = str(cfg.tapip3d.get("intrinsics_source", "depth_intrinsics_original"))
    if intrinsics_source not in meta:
        raise KeyError(f"{meta_path} does not contain intrinsics source {intrinsics_source!r}")

    intrinsics = _intrinsics_dict_to_matrix(meta[intrinsics_source])
    depth_scale = float(meta.get("depth_scale", 1.0))
    return intrinsics, depth_scale, meta


def project_to_2d(pts_3d, intrinsics):
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    z = np.maximum(pts_3d[:, 2], 1e-5)
    x2d = (pts_3d[:, 0] * fx / z) + cx
    y2d = (pts_3d[:, 1] * fy / z) + cy
    return np.stack([x2d, y2d], axis=-1).astype(int)


def _load_sample_points(ep_path: Path, cfg) -> np.ndarray:
    candidates = cfg.tapip3d.get("sample_candidates", [])
    for rel_path in candidates:
        path = ep_path / rel_path
        if path.exists():
            data = np.load(path, allow_pickle=True)
            if hasattr(data, "item"):
                data = data.item()
            if isinstance(data, dict) and "query_points_2d" in data:
                return data["query_points_2d"]
            return data
    tried = ", ".join(str(ep_path / p) for p in candidates)
    raise FileNotFoundError(f"Missing TAPIP3D sample points. Tried: {tried}")


def _lift_2d_to_3d(points_2d: np.ndarray, depth_0_meters: np.ndarray, intrinsics: np.ndarray) -> np.ndarray:
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    pts_3d_cam = []
    for p in points_2d:
        x, y = int(p[0]), int(p[1])
        if not (0 <= y < depth_0_meters.shape[0] and 0 <= x < depth_0_meters.shape[1]):
            continue
        d = depth_0_meters[y, x]
        if d <= 0 or np.isnan(d):
            continue
        z_c = float(d)
        x_c = (x - cx) * z_c / fx
        y_c = (y - cy) * z_c / fy
        pts_3d_cam.append([x_c, y_c, z_c])
    pts_3d_cam = np.array(pts_3d_cam, dtype=np.float32)
    if len(pts_3d_cam) == 0:
        raise ValueError("Depth is invalid for all sampled points; no valid 3D query points.")
    return pts_3d_cam


def _load_model(cfg, device: str):
    _add_tapip3d_to_path()
    from utils.inference_utils import load_model

    model = load_model(str(cfg.tapip3d.checkpoint))
    model.to(device)
    return model


def process_episode(ep_path: str | Path, cfg, model, device: str) -> bool:
    _add_tapip3d_to_path()
    from utils.inference_utils import inference

    ep_path = Path(ep_path)
    rgb_path = ep_path / "rgb"
    depth_path = ep_path / "depth"
    if not rgb_path.exists() or not depth_path.exists():
        raise FileNotFoundError(f"Missing rgb/depth zarr under {ep_path}")

    out_dir = ep_path / "point_tracking"
    viz_dir = ep_path / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir.mkdir(parents=True, exist_ok=True)
    npz_save_path = out_dir / "tapip3d_result.npz"
    if npz_save_path.exists() and not bool(cfg.runtime.overwrite):
        print(f"[tapip3d] skip existing {ep_path.name}")
        return True

    intrinsics_base, depth_scale, meta = load_episode_camera_params(ep_path, cfg)
    video_data = zarr.open(str(rgb_path), mode="r")[:]
    depth_raw = zarr.open(str(depth_path), mode="r")[:]
    depth_data = depth_raw.astype(np.float32) * depth_scale
    pts_2d = _load_sample_points(ep_path, cfg)

    T, H, W, _ = video_data.shape
    pts_3d_cam = _lift_2d_to_3d(pts_2d, depth_data[0], intrinsics_base)

    query_points = np.zeros((len(pts_3d_cam), 4), dtype=np.float32)
    query_points[:, 0] = 0
    query_points[:, 1:] = pts_3d_cam

    intrinsics_seq = np.tile(intrinsics_base, (T, 1, 1)).astype(np.float32)
    extrinsics_seq = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))

    video_t = (torch.from_numpy(video_data).permute(0, 3, 1, 2).float() / 255.0).to(device)
    depths_t = torch.from_numpy(depth_data).float().to(device)
    intrinsics_t = torch.from_numpy(intrinsics_seq).float().to(device)
    extrinsics_t = torch.from_numpy(extrinsics_seq).float().to(device)
    query_point_t = torch.from_numpy(query_points).float().to(device)

    autocast_enabled = device.startswith("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        coords, visibs = inference(
            model=model,
            video=video_t,
            depths=depths_t,
            intrinsics=intrinsics_t,
            extrinsics=extrinsics_t,
            query_point=query_point_t,
            num_iters=int(cfg.tapip3d.num_iters),
            grid_size=int(cfg.tapip3d.support_grid_size),
        )

    coords = coords.cpu().numpy()
    visibs = visibs.cpu().numpy()
    np.savez_compressed(
        npz_save_path,
        video=video_data,
        depths=depth_data,
        intrinsics=intrinsics_seq,
        extrinsics=extrinsics_seq,
        coords=coords,
        visibs=visibs,
        query_points=query_points,
        depth_scale=np.array(depth_scale, dtype=np.float32),
        intrinsics_source=np.array(str(cfg.tapip3d.get("intrinsics_source", "depth_intrinsics_original"))),
        depth_alignment=np.array(str(meta.get("depth_alignment", ""))),
    )

    mp4_path = viz_dir / "tracking_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(mp4_path), fourcc, 10, (W, H))
    colors = np.random.randint(0, 255, size=(coords.shape[1], 3), dtype=np.uint8)
    visib_threshold = float(cfg.tapip3d.visib_threshold)

    for t in range(T):
        frame_bgr = cv2.cvtColor(video_data[t], cv2.COLOR_RGB2BGR)
        pts_2d_t = project_to_2d(coords[t], intrinsics_base)
        for p_idx in range(coords.shape[1]):
            if visibs[t, p_idx] > visib_threshold:
                x2d, y2d = pts_2d_t[p_idx]
                if 0 <= x2d < W and 0 <= y2d < H:
                    color = tuple(int(c) for c in colors[p_idx])
                    cv2.circle(frame_bgr, (x2d, y2d), radius=4, color=color, thickness=-1)
                    cv2.circle(frame_bgr, (x2d, y2d), radius=4, color=(0, 0, 0), thickness=1)
        out.write(frame_bgr)
    out.release()

    print(f"[tapip3d] {ep_path.name}: tracked {coords.shape[1]} points across {T} frames")
    return True


def run(cfg) -> None:
    device = _device(cfg)
    print(f"[tapip3d] loading model on {device}: {cfg.tapip3d.checkpoint}")
    model = _load_model(cfg, device)

    failed = []
    for ep_path in iter_processed_episodes(cfg.paths.processed_root, cfg.runtime.episodes):
        try:
            process_episode(ep_path, cfg, model, device)
        except Exception as exc:
            print(f"[tapip3d] failed {Path(ep_path).name}: {exc}")
            failed.append((Path(ep_path).name, str(exc)))
    if failed:
        print(f"[tapip3d] failed {len(failed)} episodes")
