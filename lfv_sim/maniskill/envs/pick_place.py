from __future__ import annotations

import numpy as np
import sapien
import torch

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from .pouring import (
    LFVTabletopBaseEnv,
    YCB_SAFE_INITIAL_Z,
    _build_ycb_actor,
    _reset_dynamic_actor,
    _tabletop_z,
    _wxyz_from_euler,
)


@register_env("LFVPickBananaPlate-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class LFVPickBananaPlateEnv(LFVTabletopBaseEnv):
    """Tabletop banana-to-plate scene matching the LFV pick-and-place data roles."""

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.banana = _build_ycb_actor(
            self.scene,
            "011_banana",
            "banana",
            sapien.Pose(p=[-0.10, -0.20, YCB_SAFE_INITIAL_Z]),
            "dynamic",
        )
        self.plate = _build_ycb_actor(
            self.scene,
            "029_plate",
            "plate",
            sapien.Pose(p=[-0.10, 0.20, YCB_SAFE_INITIAL_Z]),
            "kinematic",
            scale_multiplier=0.72,
        )

    def _after_reconfigure(self, options: dict):
        self.banana_table_z = _tabletop_z(self.banana, fallback=0.035)
        self.plate_table_z = _tabletop_z(self.plate, fallback=0.018)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            options = options or {}
            layout = options.get("layout", {})
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._initialize_robot()

            banana_xy = layout.get("banana_xy", [-0.10, -0.20])
            banana_yaw = float(layout.get("banana_yaw", np.pi / 2))
            banana_position = torch.tensor(
                [[float(banana_xy[0]), float(banana_xy[1]), self.banana_table_z]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            banana_rotation = torch.tensor(
                _wxyz_from_euler(0.0, 0.0, banana_yaw),
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            _reset_dynamic_actor(
                self.banana,
                Pose.create_from_pq(banana_position, banana_rotation),
                batch_size,
                self.device,
            )

            plate_xy = layout.get("plate_xy", [-0.10, 0.20])
            plate_yaw = float(layout.get("plate_yaw", 0.0))
            plate_position = torch.tensor(
                [[float(plate_xy[0]), float(plate_xy[1]), self.plate_table_z]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            plate_rotation = torch.tensor(
                _wxyz_from_euler(0.0, 0.0, plate_yaw),
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            self.plate.set_pose(Pose.create_from_pq(plate_position, plate_rotation))

    def evaluate(self):
        banana_position = self.banana.pose.p
        plate_position = self.plate.pose.p
        over_plate = torch.linalg.norm(
            (banana_position - plate_position)[:, :2], axis=1
        ) < 0.08
        return {
            "success": over_plate,
            "is_over_plate": over_plate,
            "is_grasped": self.agent.is_grasping(self.banana),
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "is_grasped": info["is_grasped"],
            "target_pos": self.plate.pose.p,
        }
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.banana.pose.raw_pose,
                target_pose=self.plate.pose.raw_pose,
            )
        return obs
