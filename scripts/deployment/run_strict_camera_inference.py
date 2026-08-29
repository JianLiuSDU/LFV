#!/usr/bin/env python3
"""Strict single-image LFV inference using the repository's original stages.

Unlike the earlier smoke script, this entry point has no GrabCut or synthetic
mask fallback.  It calls the existing Grounding-DINO detector, SAM2 box
segmenter, AffCorrs+FGW transfer, the fixed 256-point sampler and Full64 joint
functional-motion checkpoint. A small top-down contact-pair selector is only the final grasp
instantiation requested by the deployment contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image

from lfv.affordance_transfer.app import run_transfer
from lfv.deployment.model_backend import FunctionalMotionDirectBackend
from lfv.deployment.partial_grasp import build_contact_pair_hypotheses
from lfv.inference.functional_motion.two_stage_pouring import sample_heat_point_cloud, sample_mask_point_cloud
from lfv.pipeline.dino_bbox import _load_model as load_grounding_dino, get_object_bbox
from lfv.visualization.contact_pair import save_contact_pair_ply, save_partial_grasp_overlay
from lfv.utils.config import load_config


def _find_images(root: Path) -> tuple[Path, Path]:
    rgb = depth = None
    for path in sorted(root.iterdir()):
        try:
            mode = Image.open(path).mode
        except Exception:
            continue
        if mode == "RGB" and rgb is None:
            rgb = path
        elif mode != "RGB" and depth is None:
            depth = path
    if rgb is None or depth is None:
        raise FileNotFoundError("Input folder must contain one RGB image and one depth image")
    return rgb, depth


def _camera_intrinsics(path: Path) -> tuple[np.ndarray, float]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    camera = cfg["camera"]
    return np.asarray(camera["color"]["camera_matrix"], dtype=np.float32), float(camera["depth"].get("depth_scale_m_per_unit", 0.001))


def _sam_mask(predictor, rgb: np.ndarray, box: np.ndarray, device: str) -> tuple[np.ndarray, float]:
    predictor.set_image(rgb)
    enabled = str(device).startswith("cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=enabled):
        masks, scores, _ = predictor.predict(point_coords=None, point_labels=None, box=np.asarray(box, dtype=np.float32)[None], multimask_output=True)
    i = int(np.argmax(scores))
    return np.asarray(masks[i], dtype=bool), float(scores[i])


def _save_detection_overlay(rgb: np.ndarray, boxes: dict[str, np.ndarray], masks: dict[str, np.ndarray], out: Path) -> None:
    canvas = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    colors = {"cup": (0, 220, 0), "bowl": (255, 80, 0)}
    for name, box in boxes.items():
        x0, y0, x1, y1 = np.asarray(box).astype(int)
        color = colors[name]
        cv2.rectangle(canvas, (x0, y0), (x1, y1), color, 2)
        cv2.putText(canvas, name, (x0, max(20, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        mask = masks.get(name)
        if mask is not None:
            overlay = np.zeros_like(canvas); overlay[mask] = color
            canvas = np.where(mask[..., None], (0.55 * canvas + 0.45 * overlay).astype(np.uint8), canvas)
    out.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(out), canvas)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=Path, default=Path("/home/users1/ljian/LFV_ex/cup_pouring/ex_1/input"))
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--perception-config", type=Path, default=Path("configs/pipeline/hand_pouring.yaml"))
    p.add_argument("--transfer-config", type=Path, default=Path("configs/affordance_transfer/episode0_to_ace_red_mug_fgw_k64.yaml"))
    p.add_argument("--sam2-root", type=Path, default=Path("/home/users1/ljian/sam2"))
    p.add_argument("--device", default="cpu", help="shared perception/Stage1 device")
    p.add_argument("--stage2-device", default="cpu")
    p.add_argument("--stage2-checkpoint", type=Path, default=Path("/home/users1/ljian/lfv_runs/stage2/ablation_stage_aware/a3b_gated_phase_tokens/checkpoints/best.pt"))
    p.add_argument("--dino-weights", type=Path, default=Path("/home/users1/ljian/LFV/third_party/dinov2_weights/dinov2_vits14_pretrain.pth"))
    p.add_argument("--skip-stage2", action="store_true")
    args = p.parse_args()
    root = args.input_dir.expanduser().resolve(); out = (args.output_dir or root.parent / "strict_inference").expanduser().resolve(); out.mkdir(parents=True, exist_ok=True)
    rgb_path, depth_path = _find_images(root)
    rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
    k, depth_scale = _camera_intrinsics(root / "intrinsics.yaml")
    depth = np.asarray(Image.open(depth_path), dtype=np.float32) * depth_scale

    # Existing Grounding-DINO implementation and task prompts.
    perception_cfg = load_config(args.perception_config)
    detector_device = args.device if not (args.device.startswith("cuda") and not torch.cuda.is_available()) else "cpu"
    processor, detector = load_grounding_dino(perception_cfg, detector_device)
    prompts = {"cup": str(perception_cfg.objects.affordance.prompt), "bowl": str(perception_cfg.objects.target.prompt)}
    boxes = {name: get_object_bbox(rgb, prompt, detector_device, processor, detector, float(perception_cfg.object.box_threshold), float(perception_cfg.object.text_threshold)) for name, prompt in prompts.items()}
    np.save(out / "cup_bbox.npy", boxes["cup"]); np.save(out / "bowl_bbox.npy", boxes["bowl"])

    # Existing SAM2 implementation, with the installed official source tree
    # supplied explicitly rather than changing the segmentation algorithm.
    sys.path.insert(0, str(args.sam2_root.expanduser().resolve()))
    from lfv.pipeline.sam2_mask import _load_predictor
    # SAM2's Hydra loader resolves config names inside the imported ``sam2``
    # package; passing an absolute filesystem path produces a malformed
    # ``home/...`` config name.  The source tree is inserted above, so retain
    # the package-relative name expected by the original helper.
    perception_cfg.sam2.model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    sam_device = detector_device
    predictor = _load_predictor(perception_cfg, sam_device)
    masks: dict[str, np.ndarray] = {}; sam_scores: dict[str, float] = {}
    for name, box in boxes.items():
        masks[name], sam_scores[name] = _sam_mask(predictor, rgb, box, sam_device)
        Image.fromarray((masks[name] * 255).astype(np.uint8)).save(out / f"{name}_mask.png")
    _save_detection_overlay(rgb, boxes, masks, out / "detection_segmentation_overlay.png")

    # Stage 1: exact existing AffCorrs+FGW app; only target snapshot and
    # runtime device are changed for this camera observation.
    snapshot = out / "camera_snapshot.npz"
    np.savez_compressed(snapshot, rgb=rgb, depth_m=depth, cup_mask=masks["cup"], bowl_mask=masks["bowl"], intrinsic_cv=k, T_world_to_camera=np.eye(4, dtype=np.float32), T_object_to_world=np.eye(4, dtype=np.float32))
    transfer_cfg = yaml.safe_load(args.transfer_config.read_text(encoding="utf-8"))
    transfer_cfg["target"]["snapshot_path"] = str(snapshot); transfer_cfg["target"]["mask_key"] = "cup_mask"; transfer_cfg["target"]["part_mask_key"] = "cup_mask"; transfer_cfg.setdefault("runtime", {})["device"] = args.device
    transfer_run = run_transfer(transfer_cfg, output_dir_override=out / "stage1_transfer", device_override=args.device)
    heat = np.asarray(transfer_run["result"].target_heatmap, dtype=np.float32)

    # The repository's fixed Stage 2 contract is used for both grasp evidence
    # and model input: 256 manipulated points and 256 reference points. The
    # trajectory decoder returns the trained Full64 sequence.
    points256, pixels256, heat256 = sample_heat_point_cloud(heat, masks["cup"], depth, k, 256)
    target256, target_pixels256 = sample_mask_point_cloud(masks["bowl"], depth, k, 256)
    hypotheses = build_contact_pair_hypotheses(points256, heat256, top_k=8)
    if not hypotheses:
        raise RuntimeError("No top-down contact pair was found in the transferred 256-point heat cloud")
    selected = hypotheses[0]
    trajectory = None
    if not args.skip_stage2:
        motion = FunctionalMotionDirectBackend(checkpoint=args.stage2_checkpoint, dino_weights=args.dino_weights, device=args.stage2_device, seed=42, num_goals=1, num_trajectories=1)
        pred = motion.predict(workdir=out / "stage2_motion", rgb=rgb, depth_m=depth, cup_mask=masks["cup"], bowl_mask=masks["bowl"], intrinsic_cv=k, heatmap=heat)
        trajectory = np.asarray(pred.object_trajectory_camera, dtype=np.float32)
        attachment = np.linalg.inv(pred.goal_camera) @ selected.tcp_camera
        tcp_trajectory = trajectory @ attachment
        np.savez_compressed(out / "motion_prediction.npz", goal_camera=pred.goal_camera, object_trajectory_camera=trajectory, tcp_trajectory_camera=tcp_trajectory, attachment_camera=attachment)
    else:
        tcp_trajectory = None
    save_partial_grasp_overlay(rgb, k, heat, masks["cup"], selected.first_contact_camera, selected.second_contact_camera, selected.tcp_camera, out / "inference_overlay.png", title="Strict DINO+SAM2+AffCorrs/FGW+Stage2", trajectory_camera=tcp_trajectory)
    save_contact_pair_ply(points256, heat256, selected.first_contact_camera, selected.second_contact_camera, selected.tcp_camera, out / "camera_grasp_trajectory.ply", trajectory_camera=tcp_trajectory)
    np.savez_compressed(out / "camera_plan.npz", tcp_camera=selected.tcp_camera, first_contact_camera=selected.first_contact_camera, second_contact_camera=selected.second_contact_camera, object_trajectory_camera=np.empty((0, 4, 4), np.float32) if trajectory is None else trajectory, tcp_trajectory_camera=np.empty((0, 4, 4), np.float32) if tcp_trajectory is None else tcp_trajectory, manipulated_points_stage1=points256, manipulated_pixels_stage1=pixels256, manipulated_heat_stage1=heat256, target_points_stage1=target256, target_pixels_stage1=target_pixels256, intrinsic_cv=k)
    report = {"input_rgb": str(rgb_path), "input_depth": str(depth_path), "detector": {"implementation": "lfv.pipeline.dino_bbox Grounding-DINO", "prompts": prompts, "boxes": {k: v.tolist() for k, v in boxes.items()}}, "segmenter": {"implementation": "lfv.pipeline.sam2_mask SAM2", "scores": sam_scores, "checkpoint": str(perception_cfg.sam2.checkpoint)}, "stage1": {"implementation": "run_transfer + AffCorrsFGWContactTransferPipeline", "source": transfer_cfg["source"], "accepted": bool(transfer_run["result"].accepted), "confidence": transfer_run["result"].confidence}, "sampling": {"stage1_manipulated": list(points256.shape), "stage1_target": list(target256.shape), "stage2_manipulated": [256, 3], "stage2_target": [256, 3], "trajectory": [64, 9], "implementation": "lfv.inference.functional_motion.two_stage_pouring"}, "grasp": selected.as_dict(), "trajectory_steps": 0 if trajectory is None else int(len(trajectory)), "output_dir": str(out)}
    (out / "strict_inference_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=lambda x: None), encoding="utf-8")
    print(json.dumps({"output": str(out), "stage1_accepted": report["stage1"]["accepted"], "trajectory_steps": report["trajectory_steps"], "files": ["detection_segmentation_overlay.png", "stage1_transfer/transfer_summary.png", "inference_overlay.png", "camera_plan.npz", "camera_grasp_trajectory.ply", "strict_inference_report.json"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
