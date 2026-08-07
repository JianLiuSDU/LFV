from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill import ASSET_DIR
from mani_skill.agents.robots import Panda
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs import Pose
from mani_skill.utils.structs.types import Array, SimConfig

from lfv_sim.maniskill.robots import PandaDrawerFinger, PandaLongFinger


YCB_ASSET_INFO = ASSET_DIR / "assets/mani_skill2_ycb/info_pick_v0.json"
TABLE_SURFACE_Z = 0.0
YCB_TABLE_CLEARANCE = 0.008
YCB_SAFE_INITIAL_Z = 0.30


def _wxyz_from_euler(ai: float, aj: float, ak: float) -> np.ndarray:
    return np.asarray(euler2quat(ai, aj, ak, axes="sxyz"), dtype=np.float32)


def _require_ycb_assets() -> None:
    if not YCB_ASSET_INFO.exists():
        raise FileNotFoundError(
            f"Missing ManiSkill YCB assets at {YCB_ASSET_INFO}. "
            "Run `python -m mani_skill.utils.download_asset ycb -y`."
        )


def _build_ycb_actor(
    scene,
    model_id: str,
    name: str,
    initial_pose,
    body_type: str,
    scale_multiplier: float = 1.0,
):
    _require_ycb_assets()
    if abs(float(scale_multiplier) - 1.0) < 1e-8:
        builder = actors.get_actor_builder(scene, id=f"ycb:{model_id}")
    else:
        metadata = json.loads(YCB_ASSET_INFO.read_text(encoding="utf-8"))[model_id]
        scale = float(metadata.get("scales", [1.0])[0]) * float(scale_multiplier)
        density = float(metadata.get("density", 1000.0))
        model_dir = YCB_ASSET_INFO.parent / "models" / model_id
        builder = scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename=str(model_dir / "collision.ply"),
            scale=[scale] * 3,
            density=density,
        )
        builder.add_visual_from_file(
            filename=str(model_dir / "textured.obj"), scale=[scale] * 3
        )
    builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        return builder.build(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    raise ValueError(f"Unsupported actor body type {body_type!r}")


def _build_mesh_actor(
    scene,
    asset: dict,
    name: str,
    initial_pose,
    body_type: str,
):
    visual_file = Path(asset["visual_file"]).expanduser().resolve()
    collision_files = [
        Path(path).expanduser().resolve() for path in asset["collision_files"]
    ]
    if not visual_file.is_file():
        raise FileNotFoundError(f"Missing custom cup visual mesh: {visual_file}")
    missing = [str(path) for path in collision_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing custom cup collision meshes: {missing}")
    if not collision_files:
        raise ValueError("A custom cup requires at least one convex collision mesh")

    scale_value = float(asset.get("scale", 1.0))
    scale = [scale_value] * 3
    density = float(asset.get("density", 1000.0))
    base_color = np.asarray(
        asset.get("base_color", [0.08, 0.18, 0.65, 1.0]),
        dtype=np.float32,
    )
    if base_color.shape != (4,):
        raise ValueError(f"base_color must contain RGBA values, got {base_color}")
    render_material = sapien.render.RenderMaterial(
        base_color=base_color,
        roughness=float(asset.get("roughness", 0.55)),
        metallic=float(asset.get("metallic", 0.0)),
    )

    builder = scene.create_actor_builder()
    builder.add_visual_from_file(
        filename=str(visual_file),
        scale=scale,
        material=render_material,
    )
    for collision_file in collision_files:
        builder.add_convex_collision_from_file(
            filename=str(collision_file),
            scale=scale,
            density=density,
        )
    builder.set_initial_pose(initial_pose)
    if body_type == "dynamic":
        return builder.build(name=name)
    if body_type == "kinematic":
        return builder.build_kinematic(name=name)
    raise ValueError(f"Unsupported actor body type {body_type!r}")


def _tabletop_z(actor, fallback: float) -> float:
    mesh = actor.get_first_collision_mesh(to_world_frame=False)
    if mesh is None:
        return fallback
    return float(TABLE_SURFACE_Z - mesh.bounding_box.bounds[0, 2] + YCB_TABLE_CLEARANCE)


def _reset_dynamic_actor(actor, pose: Pose, batch_size: int, device) -> None:
    actor.set_pose(pose)
    actor.set_linear_velocity(torch.zeros((batch_size, 3), device=device))
    actor.set_angular_velocity(torch.zeros((batch_size, 3), device=device))


class LFVTabletopBaseEnv(BaseEnv):
    SUPPORTED_ROBOTS = ["panda", "panda_long_finger", "panda_drawer_finger"]
    agent: Union[Panda, PandaLongFinger, PandaDrawerFinger]

    sensor_camera_eye = [0.45, 0.0, 0.53]
    sensor_camera_target = [-0.10, 0.0, 0.03]
    opposite_camera_eye = [-0.58, 0.38, 0.82]
    opposite_camera_target = [0.02, 0.02, 0.06]
    record_camera_eye = [0.72, -0.60, 0.72]
    record_camera_target = [-0.12, 0.02, 0.10]
    render_camera_eye = [0.54, 0.0, 0.64]
    render_camera_target = [-0.10, 0.0, 0.06]

    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.01, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig()

    @property
    def _default_sensor_configs(self):
        return [
            CameraConfig(
                "base_camera",
                pose=sapien_utils.look_at(self.sensor_camera_eye, self.sensor_camera_target),
                width=640,
                height=480,
                fov=0.92,
                near=0.01,
                far=10,
            ),
            CameraConfig(
                "opposite_camera",
                pose=sapien_utils.look_at(self.opposite_camera_eye, self.opposite_camera_target),
                width=640,
                height=480,
                fov=0.86,
                near=0.01,
                far=10,
            ),
            CameraConfig(
                "record_camera",
                pose=sapien_utils.look_at(self.record_camera_eye, self.record_camera_target),
                width=960,
                height=540,
                fov=1.02,
                near=0.01,
                far=10,
            ),
        ]

    @property
    def _default_human_render_camera_configs(self):
        return CameraConfig(
            "render_camera",
            pose=sapien_utils.look_at(self.render_camera_eye, self.render_camera_target),
            width=1280,
            height=720,
            fov=0.86,
            near=0.01,
            far=10,
        )

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))

    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self,
            robot_init_qpos_noise=self.robot_init_qpos_noise,
        )
        self.table_scene.build()

    def _initialize_robot(self):
        qpos = np.array(
            [0.0, 0, 0, -np.pi * 2 / 3, 0, np.pi * 2 / 3, np.pi / 4, 0.04, 0.04]
        )
        qpos[:-2] += self._episode_rng.normal(
            0,
            self.robot_init_qpos_noise,
            len(qpos) - 2,
        )
        self.agent.reset(qpos)
        self.agent.robot.set_root_pose(sapien.Pose([-0.615, 0, 0]))

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        return torch.zeros(self.num_envs, device=self.device)

    def compute_normalized_dense_reward(self, obs: Any, action: Array, info: dict):
        return self.compute_dense_reward(obs=obs, action=action, info=info)


@register_env("LFVPourCupBowl-v1", max_episode_steps=100, asset_download_ids=["ycb"])
class LFVPourCupBowlEnv(LFVTabletopBaseEnv):
    """LFV pouring scene with a configurable dynamic mug and YCB bowl."""

    def __init__(self, *args, cup_asset: dict | None = None, **kwargs):
        self.cup_asset = dict(
            cup_asset or {"kind": "ycb", "model_id": "025_mug"}
        )
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        cup_kind = str(self.cup_asset.get("kind", "ycb"))
        if cup_kind == "ycb":
            self.cup = _build_ycb_actor(
                self.scene,
                str(self.cup_asset.get("model_id", "025_mug")),
                "cup",
                sapien.Pose(p=[-0.12, 0.085, YCB_SAFE_INITIAL_Z]),
                "dynamic",
            )
        elif cup_kind == "mesh":
            self.cup = _build_mesh_actor(
                self.scene,
                self.cup_asset,
                "cup",
                sapien.Pose(p=[-0.12, 0.085, YCB_SAFE_INITIAL_Z]),
                "dynamic",
            )
        else:
            raise ValueError(
                f"Unsupported cup asset kind {cup_kind!r}; expected 'ycb' or 'mesh'"
            )
        self.bowl = _build_ycb_actor(
            self.scene,
            "024_bowl",
            "bowl",
            sapien.Pose(p=[0.09, -0.045, YCB_SAFE_INITIAL_Z]),
            "kinematic",
        )

    def _after_reconfigure(self, options: dict):
        self.cup_table_z = _tabletop_z(self.cup, fallback=0.05)
        self.bowl_table_z = _tabletop_z(self.bowl, fallback=0.04)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            options = options or {}
            layout = options.get("layout", {})
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._initialize_robot()

            cup_xy = layout.get("cup_xy", [-0.12, 0.09])
            cup_position = torch.tensor(
                [[float(cup_xy[0]), float(cup_xy[1]), self.cup_table_z]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            cup_yaw = float(layout.get("cup_yaw", -np.pi / 2))
            cup_rotation = torch.tensor(
                _wxyz_from_euler(0, 0, cup_yaw),
                device=self.device,
            ).repeat(batch_size, 1)
            _reset_dynamic_actor(
                self.cup,
                Pose.create_from_pq(cup_position, cup_rotation),
                batch_size,
                self.device,
            )

            bowl_xy = layout.get("bowl_xy", [0.08, -0.05])
            bowl_position = torch.tensor(
                [[float(bowl_xy[0]), float(bowl_xy[1]), self.bowl_table_z]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            bowl_rotation = torch.tensor(
                [1, 0, 0, 0],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            self.bowl.set_pose(Pose.create_from_pq(bowl_position, bowl_rotation))

    def evaluate(self):
        cup_position = self.cup.pose.p
        bowl_position = self.bowl.pose.p
        above = torch.linalg.norm((cup_position - bowl_position)[:, :2], axis=1) < 0.06
        high = cup_position[:, 2] > bowl_position[:, 2] + 0.08
        return {
            "success": above & high,
            "is_cup_above_bowl": above,
            "is_grasped": self.agent.is_grasping(self.cup),
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "is_grasped": info["is_grasped"],
            "target_pos": self.bowl.pose.p,
        }
        if "state" in self.obs_mode:
            obs.update(obj_pose=self.cup.pose.raw_pose, target_pose=self.bowl.pose.raw_pose)
        return obs
