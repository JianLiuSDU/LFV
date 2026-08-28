#!/usr/bin/env python3
"""Offline camera RGB-D -> LFV grasp/trajectory artifact."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from lfv.deployment.camera_plan_pipeline import CameraToPlanPipeline
from lfv.deployment.grasp_backend import ExternalGraspNetBackend, NPZGraspBackend
from lfv.deployment.model_backend import ExternalMotionBackend, LegacyPouringBackend
from lfv.deployment.transfer_backend import ExternalHeatBackend, LocalAffCorrsBackend, PrecomputedHeatBackend
from lfv.geometry.sam3d_completion import NPZCompletionBackend, SAM3DSubprocessBackend, VisibleOnlyCompletionBackend
from lfv.perception.backends import ExternalMaskBackend, PrecomputedMaskBackend


def _section(cfg: dict, name: str) -> dict:
    value = cfg.get(name, {})
    return value if isinstance(value, dict) else {}


def build_pipeline(cfg: dict) -> CameraToPlanPipeline:
    perception_cfg = _section(cfg, "perception")
    if perception_cfg.get("backend", "precomputed") == "external":
        perception = ExternalMaskBackend(perception_cfg["command"], perception_cfg.get("output_name", "masks.npz"))
    else:
        perception = PrecomputedMaskBackend(perception_cfg.get("cup_mask"), perception_cfg.get("bowl_mask"))
    transfer_cfg = _section(cfg, "transfer")
    if transfer_cfg.get("backend", "precomputed") == "affcorrs":
        transfer = LocalAffCorrsBackend(transfer_cfg["source_rgb"], transfer_cfg["source_mask"], transfer_cfg["source_heatmap"], transfer_cfg["dino_weights"], dino_model=transfer_cfg.get("dino_model", "vit_small_patch14_dinov2"), device=transfer_cfg.get("device", "cuda"), config=transfer_cfg.get("config", {}))
    elif transfer_cfg.get("backend", "precomputed") == "external":
        transfer = ExternalHeatBackend(transfer_cfg["command"], transfer_cfg.get("output_name", "transfer_result.npz"))
    else:
        transfer = PrecomputedHeatBackend(transfer_cfg["path"])
    completion_cfg = _section(cfg, "completion")
    backend = completion_cfg.get("backend", "visible_only")
    if backend == "npz":
        completion = NPZCompletionBackend(completion_cfg["path"])
    elif backend == "sam3d":
        completion = SAM3DSubprocessBackend(completion_cfg["python_executable"], completion_cfg["repo"], completion_cfg["config"], seed=completion_cfg.get("seed", 42))
    else:
        completion = VisibleOnlyCompletionBackend(completion_cfg.get("max_depth_m", 5.0))
    grasp_cfg = _section(cfg, "grasp")
    grasp = None
    if grasp_cfg.get("backend") == "npz":
        grasp = NPZGraspBackend(grasp_cfg["path"], preferred_approach_camera=tuple(grasp_cfg.get("preferred_approach_camera", [0.0, -1.0, 0.0])), topdown_weight=grasp_cfg.get("topdown_weight", 0.5))
    elif grasp_cfg.get("backend") == "external":
        grasp = ExternalGraspNetBackend(grasp_cfg["command"], grasp_cfg.get("output_name", "grasps.npz"), preferred_approach_camera=tuple(grasp_cfg.get("preferred_approach_camera", [0.0, -1.0, 0.0])), topdown_weight=grasp_cfg.get("topdown_weight", 0.5))
    motion_cfg = _section(cfg, "motion")
    motion = None
    if motion_cfg.get("backend") == "legacy_pouring":
        motion = LegacyPouringBackend(model_repo=motion_cfg["model_repo"], goal_checkpoint=motion_cfg["goal_checkpoint"], trajectory_checkpoint=motion_cfg["trajectory_checkpoint"], language_embedding=motion_cfg.get("language_embedding"), python_executable=motion_cfg.get("python_executable", "python"), steps_script=motion_cfg.get("steps_script"), seed=motion_cfg.get("seed", 42), device=motion_cfg.get("device", "cuda:0"))
    elif motion_cfg.get("backend") == "external":
        motion = ExternalMotionBackend(motion_cfg["command"], motion_cfg.get("output_name", "motion_prediction.npz"))
    return CameraToPlanPipeline(perception=perception, transfer=transfer, completion=completion, grasp=grasp, motion=motion, allow_fallback_grasp=bool(cfg.get("allow_fallback_grasp", False)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    input_dir = args.input_dir or cfg.get("input_dir", "/home/users1/ljian/LFV_ex/cup_pouring/ex_1/input")
    output_dir = args.output_dir or cfg.get("output_dir", str(Path(input_dir).parent))
    result = build_pipeline(cfg).run(input_dir, output_dir)
    print(f"Saved camera plan to {Path(output_dir).resolve() / 'camera_plan.npz'}")
    print(result.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
