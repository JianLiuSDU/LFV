#!/usr/bin/env python3
"""Run the complete offline LFV inference on ``cup_pouring/ex_1/input``.

The input folder contains the camera RGB image, a 16-bit aligned depth image,
and the RealSense YAML intrinsics.  RGB object masks are obtained with a
deterministic GrabCut ROI fallback (replaceable by the existing SAM2/DINO
external backend).  Stage 1 performs the repository AffCorrs/FGW transfer;
the legacy Stage 2 checkpoints then generate the camera-frame trajectory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from lfv.affordance_transfer.app import run_transfer
from lfv.deployment.partial_grasp import build_contact_pair_hypotheses
from lfv.deployment.model_backend import LegacyPouringBackend
from lfv.geometry.sam3d_completion import backproject_mask
from lfv.visualization.contact_pair import save_contact_pair_ply, save_partial_grasp_overlay


def _images(root: Path) -> tuple[Path, Path]:
    rgb_path = depth_path = None
    for p in sorted(root.glob("*")):
        try:
            mode = Image.open(p).mode
        except Exception:
            continue
        if mode == "RGB" and rgb_path is None:
            rgb_path = p
        elif mode != "RGB" and depth_path is None:
            depth_path = p
    if rgb_path is None or depth_path is None:
        raise FileNotFoundError("input must contain one RGB image and one depth image")
    return rgb_path, depth_path


def _intrinsic(path: Path) -> np.ndarray:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    color = data["camera"]["color"]
    return np.asarray(color["camera_matrix"], dtype=np.float32)


def _grabcut_mask(rgb: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    h, w = rgb.shape[:2]
    x, y, rw, rh = rect
    x, y = max(0, x), max(0, y)
    rw, rh = min(w - x, rw), min(h - y, rh)
    labels = np.zeros((h, w), dtype=np.uint8)
    bgd = np.zeros((1, 65), np.float64); fgd = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), labels, (x, y, rw, rh), bgd, fgd, 8, cv2.GC_INIT_WITH_RECT)
    mask = np.isin(labels, (1, 3)).astype(np.uint8)
    # Keep the principal component inside the requested ROI; GrabCut can
    # otherwise retain small cable/table fragments near the rectangle.
    n, comp, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if n > 1:
        ids = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0 and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] > x and stats[i, cv2.CC_STAT_LEFT] < x + rw and stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] > y and stats[i, cv2.CC_STAT_TOP] < y + rh]
        if ids:
            mask = (comp == max(ids, key=lambda i: int(stats[i, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return mask.astype(bool)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("/home/users1/ljian/LFV_ex/cup_pouring/ex_1/input"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", default="cpu", help="DINO/FGW device; cpu is reproducible on the 3060 computer")
    parser.add_argument("--skip-motion", action="store_true")
    args = parser.parse_args()
    root = args.input_dir.expanduser().resolve(); out = (args.output_dir or root.parent / "inference").expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    rgb_path, depth_path = _images(root)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) * 0.001
    k = _intrinsic(root / "intrinsics.yaml")
    # The input camera layout matches the pouring protocol: cup on the left,
    # bowl on the right.  The rectangles are only initialization hints; the
    # object boundaries are obtained by GrabCut.
    cup_mask = _grabcut_mask(rgb, (70, 210, 260, 190))
    bowl_mask = _grabcut_mask(rgb, (390, 185, 190, 170))
    Image.fromarray((cup_mask * 255).astype(np.uint8)).save(out / "cup_mask.png")
    Image.fromarray((bowl_mask * 255).astype(np.uint8)).save(out / "bowl_mask.png")
    # Save a target snapshot understood by the existing Stage 1 adapters.
    snapshot = out / "camera_snapshot.npz"
    np.savez_compressed(snapshot, rgb=rgb, depth_m=depth, cup_mask=cup_mask, bowl_mask=bowl_mask, intrinsic_cv=k)
    # Build the known source episode configuration while swapping only the
    # target snapshot.  All transfer computations remain the prior AffCorrs/
    # FGW implementation.
    cfg_path = Path(__file__).resolve().parents[2] / "configs" / "affordance_transfer" / "episode0_to_ace_red_mug_fgw_k64.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    cfg["target"]["snapshot_path"] = str(snapshot)
    cfg["target"]["mask_key"] = "cup_mask"; cfg["target"]["part_mask_key"] = "cup_mask"
    cfg["runtime"]["device"] = args.device
    # RealSense holes and GrabCut's conservative silhouette can disconnect a
    # sparse RGB-D graph.  This only relaxes the first-stage graph builder; it
    # does not alter the semantic matching or heatmap output.
    cfg.setdefault("fgw", {})["edge_length_ratio"] = 20.0
    cfg["fgw"]["graph_maximum_neighbors"] = 64
    cfg["fgw"]["node_count"] = min(int(cfg["fgw"].get("node_count", 256)), 128)
    transfer_out = out / "stage1_transfer"
    transfer_run = run_transfer(cfg, output_dir_override=transfer_out, device_override=args.device)
    heat = np.asarray(transfer_run["result"].target_heatmap, dtype=np.float32)
    points, uv = backproject_mask(cup_mask, depth, k)
    heat_points = heat[uv[:, 1], uv[:, 0]]
    hypotheses = build_contact_pair_hypotheses(points, heat_points, top_k=8)
    selected = hypotheses[0]
    motion_prediction = None
    tcp_trajectory = None
    if not args.skip_motion:
        motion = LegacyPouringBackend(
            model_repo="/home/users1/ljian/object_centric_diffusion",
            goal_checkpoint="/home/users1/ljian/object_centric_diffusion/data/outputs_local_goal_pose/pouring_seed42/20260428_171300/checkpoints/epoch=0700-val_sample_goal_pos_err_cm=3.086.ckpt",
            trajectory_checkpoint="/home/users1/ljian/object_centric_diffusion/data/outputs_goal_full64/pouring_seed42/20260429_210924/checkpoints/epoch=1500.ckpt",
            language_embedding="/media/ljian/lj/data_3d/pouring/lang_emb.npy",
            python_executable="/home/users1/ljian/anaconda3/envs/sam3d-objects/bin/python",
            seed=42,
            device="cpu",
        )
        motion_prediction = motion.predict(workdir=out / "stage2_motion", rgb=rgb, depth_m=depth, cup_mask=cup_mask, bowl_mask=bowl_mask, intrinsic_cv=k, heatmap=heat)
        trajectory = np.asarray(motion_prediction.object_trajectory_camera, dtype=np.float32)
        # Reuse the attachment convention from the simulator execution code:
        # object pose × (goal-object)^-1 × grasp TCP.
        attachment = np.linalg.inv(motion_prediction.goal_camera) @ selected.tcp_camera
        tcp_trajectory = trajectory @ attachment
        np.savez_compressed(out / "motion_prediction.npz", goal_camera=motion_prediction.goal_camera, object_trajectory_camera=trajectory, tcp_trajectory_camera=tcp_trajectory, attachment_camera=attachment)
    else:
        trajectory = None
    save_partial_grasp_overlay(rgb, k, heat, cup_mask, selected.first_contact_camera, selected.second_contact_camera, selected.tcp_camera, out / "inference_overlay.png", title="LFV camera inference: heat + small gripper + TCP trajectory", trajectory_camera=tcp_trajectory)
    save_contact_pair_ply(points, heat_points, selected.first_contact_camera, selected.second_contact_camera, selected.tcp_camera, out / "camera_grasp_trajectory.ply", trajectory_camera=tcp_trajectory)
    np.savez_compressed(out / "camera_grasp_and_trajectory.npz", tcp_camera=selected.tcp_camera, first_contact_camera=selected.first_contact_camera, second_contact_camera=selected.second_contact_camera, object_trajectory_camera=np.empty((0, 4, 4), np.float32) if trajectory is None else trajectory, tcp_trajectory_camera=np.empty((0, 4, 4), np.float32) if tcp_trajectory is None else tcp_trajectory, heatmap=heat, visible_points_camera=points, visible_pixels_uv=uv, intrinsic_cv=k)
    report = {"input_rgb": str(rgb_path), "input_depth": str(depth_path), "output": str(out), "mask_backend": "grabcut_roi_fallback", "stage1": {"accepted": bool(transfer_run["result"].accepted), "confidence": transfer_run["result"].confidence, "paths": {k: str(v) for k, v in transfer_run["paths"].items()}}, "visible_points": int(len(points)), "selected_grasp": selected.as_dict(), "stage2": None if motion_prediction is None else motion_prediction.metadata}
    (out / "inference_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda x: None), encoding="utf-8")
    print(json.dumps({"output": str(out), "stage1_accepted": report["stage1"]["accepted"], "trajectory_steps": 0 if trajectory is None else int(len(trajectory)), "files": ["inference_overlay.png", "camera_grasp_and_trajectory.npz", "inference_report.json"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
