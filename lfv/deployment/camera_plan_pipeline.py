from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from lfv.deployment.grasp_backend import GraspSelection, NPZGraspBackend
from lfv.deployment.input_schema import CameraInput, load_camera_input
from lfv.deployment.model_backend import MotionPrediction
from lfv.deployment.output_schema import CameraPlanResult
from lfv.deployment.quality_checks import completion_report, trajectory_report
from lfv.geometry.sam3d_completion import CompletedObject, VisibleOnlyCompletionBackend
from lfv.perception.backends import MaskPerceptionResult, PrecomputedMaskBackend, save_masks


def _heat_to_points(points: np.ndarray, visible_points: np.ndarray, visible_heat: np.ndarray) -> np.ndarray:
    from scipy.spatial import cKDTree
    index = cKDTree(np.asarray(visible_points)).query(np.asarray(points), k=1)[1]
    return np.asarray(visible_heat, dtype=np.float32)[index]


def _fallback_grasp(points: np.ndarray, heat: np.ndarray, preferred: np.ndarray = np.array([0.0, -1.0, 0.0], dtype=np.float32)) -> GraspSelection:
    points, heat = np.asarray(points, dtype=np.float32), np.asarray(heat, dtype=np.float32).reshape(-1)
    center = (points * (heat[:, None] + 1e-3)).sum(0) / float((heat + 1e-3).sum())
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    closing = vt[0].astype(np.float32)
    approach = preferred / max(float(np.linalg.norm(preferred)), 1e-8)
    closing -= approach * float(closing @ approach)
    closing /= max(float(np.linalg.norm(closing)), 1e-8)
    lateral = np.cross(closing, approach); lateral /= max(float(np.linalg.norm(lateral)), 1e-8)
    tcp = np.eye(4, dtype=np.float32); tcp[:3, :3] = np.stack((lateral, closing, approach), -1); tcp[:3, 3] = center
    return GraspSelection(tcp, np.zeros(17, dtype=np.float32), float(heat.max()), {"backend": "geometry_fallback", "warning": "not GraspNet"})


def _project(points: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    z = np.maximum(np.asarray(points)[:, 2], 1e-6)
    return np.stack((points[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2], points[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2]), -1)


def save_camera_overlay(camera: CameraInput, perception: MaskPerceptionResult, heat: np.ndarray, grasp: GraspSelection | None, trajectory: np.ndarray | None, output: str | Path) -> None:
    canvas = cv2.cvtColor(camera.rgb, cv2.COLOR_RGB2BGR)
    canvas[perception.cup_mask] = (0.55 * canvas[perception.cup_mask] + 0.45 * np.array([255, 40, 40])).astype(np.uint8)
    canvas[perception.bowl_mask] = (0.55 * canvas[perception.bowl_mask] + 0.45 * np.array([40, 255, 40])).astype(np.uint8)
    heat = np.clip(heat, 0.0, 1.0)
    heat_color = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_JET)
    active = perception.cup_mask & (heat > 0.03)
    canvas[active] = (0.45 * canvas[active] + 0.55 * heat_color[active]).astype(np.uint8)
    if grasp is not None:
        p = _project(grasp.tcp_camera[None, :3, 3], camera.intrinsic_cv)[0].astype(int)
        cv2.drawMarker(canvas, tuple(p), (255, 255, 255), cv2.MARKER_CROSS, 24, 2)
        for axis, color in zip(np.eye(3, dtype=np.float32), ((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            end = grasp.tcp_camera[:3, 3] + grasp.tcp_camera[:3, :3] @ axis * 0.06
            q = _project(end[None], camera.intrinsic_cv)[0].astype(int)
            cv2.arrowedLine(canvas, tuple(p), tuple(q), color, 2, tipLength=0.2)
    if trajectory is not None and len(trajectory):
        pixels = _project(np.asarray(trajectory)[:, :3, 3], camera.intrinsic_cv).astype(int)
        for p0, p1 in zip(pixels[:-1], pixels[1:]):
            cv2.line(canvas, tuple(p0), tuple(p1), (0, 165, 255), 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(pixels[0]), 5, (255, 255, 0), -1)
        cv2.circle(canvas, tuple(pixels[-1]), 6, (0, 0, 255), -1)
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(output), canvas)


def save_open3d_snapshot(points: np.ndarray, heat: np.ndarray, grasp: np.ndarray | None, trajectory: np.ndarray | None, output: str | Path, *, width: int = 960, height: int = 720, python_executable: str | None = None) -> None:
    """Headless Open3D screenshot in a child process, isolating GUI crashes."""
    output = Path(output); payload_path = output.with_suffix(".open3d_input.npz"); payload = {"points": np.asarray(points, dtype=np.float64), "heat": np.clip(np.asarray(heat, dtype=np.float64), 0, 1)}
    if grasp is not None: payload["grasp"] = np.asarray(grasp, dtype=np.float64)
    if trajectory is not None and len(trajectory) > 1: payload["trajectory"] = np.asarray(trajectory, dtype=np.float64)
    np.savez_compressed(payload_path, **payload)
    helper = Path(__file__).resolve().parents[2] / "tools" / "deployment" / "render_open3d_snapshot.py"
    try:
        subprocess.run([python_executable or sys.executable, str(helper), "--input", str(payload_path), "--output", str(output), "--width", str(width), "--height", str(height)], check=True, timeout=120)
    except (subprocess.SubprocessError, OSError) as exc:
        raise RuntimeError(f"Open3D renderer failed in child process: {exc}") from exc
    finally:
        payload_path.unlink(missing_ok=True)


class CameraToPlanPipeline:
    def __init__(self, *, perception: Any, transfer: Any, completion: Any, grasp: Any | None = None, motion: Any | None = None, allow_fallback_grasp: bool = False):
        self.perception, self.transfer, self.completion, self.grasp, self.motion = perception, transfer, completion, grasp, motion
        self.allow_fallback_grasp = allow_fallback_grasp

    def run(self, input_dir: str | Path, output_dir: str | Path) -> CameraPlanResult:
        input_dir, output_dir = Path(input_dir).expanduser().resolve(), Path(output_dir).expanduser().resolve(); output_dir.mkdir(parents=True, exist_ok=True)
        camera = load_camera_input(input_dir); input_report = camera.validate()
        perception = self.perception.predict(camera.rgb, input_dir); save_masks(perception, output_dir / "masks.npz")
        heat = self.transfer.transfer(image=camera.rgb, workdir=output_dir, input_dir=input_dir, perception=perception)
        np.save(output_dir / "target_heatmap.npy", heat)
        completion: CompletedObject = self.completion.complete(camera.rgb, perception.cup_mask, camera.depth_m, camera.intrinsic_cv, output_dir / "completion")
        visible_valid = perception.cup_mask & np.isfinite(camera.depth_m) & (camera.depth_m > 1e-4) & (camera.depth_m <= 5.0)
        complete_heat = _heat_to_points(completion.complete_points_camera, completion.visible_points_camera, heat[visible_valid])
        if self.grasp is not None:
            grasp = self.grasp.select(completion.complete_points_camera, complete_heat, completion.camera_from_object, workdir=output_dir)
        elif self.allow_fallback_grasp:
            grasp = _fallback_grasp(completion.complete_points_camera, complete_heat)
        else:
            raise RuntimeError("No GraspNet backend configured; set grasp.backend or allow_fallback_grasp=true")
        prediction: MotionPrediction | None = None
        if self.motion is not None:
            prediction = self.motion.predict(workdir=output_dir / "motion", rgb=camera.rgb, depth_m=camera.depth_m, cup_mask=perception.cup_mask, bowl_mask=perception.bowl_mask, intrinsic_cv=camera.intrinsic_cv, heatmap=heat)
        object_traj = prediction.object_trajectory_camera if prediction is not None else np.stack((np.eye(4, dtype=np.float32), grasp.tcp_camera), 0)
        tcp_traj = object_traj.copy()
        # camera-frame handoff uses the attachment captured at the selected grasp.
        if prediction is not None:
            attachment = np.linalg.inv(prediction.goal_camera) @ grasp.tcp_camera
            tcp_traj = object_traj @ attachment
        result = CameraPlanResult(grasp.tcp_camera, object_traj, tcp_traj, np.array([1.0, 1.0], dtype=np.float32), {"input": input_report, "perception_source": perception.source, "completion": completion_report(completion.visible_points_camera, completion.complete_points_camera), "completion_backend": completion.metadata, "grasp": grasp.metadata, "motion": None if prediction is None else prediction.metadata, "trajectory": trajectory_report(object_traj)})
        result.validate()
        np.savez_compressed(output_dir / "camera_plan.npz", selected_grasp_camera=result.selected_grasp_camera, object_trajectory_camera=result.object_trajectory_camera, tcp_trajectory_camera=result.tcp_trajectory_camera, complete_points_camera=completion.complete_points_camera, complete_heat=complete_heat)
        save_camera_overlay(camera, perception, heat, grasp, object_traj, output_dir / "camera_overlay.png")
        try: save_open3d_snapshot(completion.complete_points_camera, complete_heat, grasp.tcp_camera, object_traj, output_dir / "open3d_snapshot.png")
        except RuntimeError as exc: (output_dir / "open3d_unavailable.txt").write_text(str(exc), encoding="utf-8")
        (output_dir / "camera_plan_report.json").write_text(json.dumps(result.report, indent=2, ensure_ascii=False, default=lambda value: None), encoding="utf-8")
        return result
