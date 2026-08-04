from __future__ import annotations

from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.utils.registration import register_env
from mani_skill.utils.structs import Pose

from .pouring import LFVTabletopBaseEnv


GREEN = np.asarray([0.08, 0.38, 0.27, 1.0], dtype=np.float32)
GREEN_DARK = np.asarray([0.055, 0.27, 0.19, 1.0], dtype=np.float32)
BLACK = np.asarray([0.015, 0.018, 0.018, 1.0], dtype=np.float32)


def _material(color: np.ndarray, roughness: float = 0.62):
    return sapien.render.RenderMaterial(
        base_color=np.asarray(color, dtype=np.float32),
        roughness=float(roughness),
        metallic=0.0,
    )


def _add_box(link, *, half_size, pose, material, density: float = 700.0) -> None:
    link.add_box_visual(
        pose=sapien.Pose(p=pose),
        half_size=half_size,
        material=material,
    )
    link.add_box_collision(
        pose=sapien.Pose(p=pose),
        half_size=half_size,
        density=float(density),
    )


def build_lfv_drawer(scene, geometry: dict):
    """Build a compact one-DOF drawer with separately segmentable links.

    Local coordinates use +X as the drawer opening direction, +Y along the
    horizontal handle, and +Z upward. The cabinet front plane is near X=0.
    """

    width = float(geometry.get("width", 0.24))
    depth = float(geometry.get("depth", 0.26))
    height = float(geometry.get("height", 0.13))
    panel = float(geometry.get("panel_thickness", 0.010))
    clearance = float(geometry.get("clearance", 0.005))
    open_limit = float(geometry.get("open_limit", 0.19))
    handle_width = float(geometry.get("handle_width", 0.115))
    handle_radius = float(geometry.get("handle_radius", 0.010))
    handle_standoff = float(geometry.get("handle_standoff", 0.040))
    handle_z = float(geometry.get("handle_z", height * 0.50))

    green = _material(GREEN)
    green_dark = _material(GREEN_DARK)
    black = _material(BLACK, roughness=0.46)

    builder = scene.create_articulation_builder()

    cabinet = builder.create_link_builder(parent=None)
    cabinet.set_name("drawer_object_cabinet")
    # Static housing: top, bottom, side walls, and back wall. The front remains
    # open so the sliding body is physically meaningful.
    _add_box(
        cabinet,
        half_size=[depth / 2, width / 2, panel / 2],
        pose=[-depth / 2, 0, panel / 2],
        material=green,
    )
    _add_box(
        cabinet,
        half_size=[depth / 2, width / 2, panel / 2],
        pose=[-depth / 2, 0, height - panel / 2],
        material=green,
    )
    for side in (-1.0, 1.0):
        _add_box(
            cabinet,
            half_size=[depth / 2, panel / 2, height / 2],
            pose=[-depth / 2, side * (width / 2 - panel / 2), height / 2],
            material=green,
        )
    _add_box(
        cabinet,
        half_size=[panel / 2, width / 2, height / 2],
        pose=[-depth + panel / 2, 0, height / 2],
        material=green_dark,
    )

    moving = builder.create_link_builder(cabinet)
    moving.set_name("drawer_object_moving_body")
    moving.set_joint_name("drawer_slide_joint")
    moving.set_joint_properties(
        type="prismatic",
        limits=[[0.0, open_limit]],
        pose_in_parent=sapien.Pose(),
        pose_in_child=sapien.Pose(),
        friction=float(geometry.get("joint_friction", 0.10)),
        damping=float(geometry.get("joint_damping", 8.0)),
    )
    # Drawer front plus a shallow open tray. The front is slightly inset into
    # the housing when q=0 to avoid a visible seam.
    front_x = -panel * 0.25
    drawer_half_width = width / 2 - panel - clearance
    drawer_floor_z = panel + clearance
    drawer_depth = depth * 0.78
    _add_box(
        moving,
        half_size=[panel / 2, drawer_half_width, (height - 2 * clearance) / 2],
        pose=[front_x, 0, height / 2],
        material=green,
    )
    _add_box(
        moving,
        half_size=[drawer_depth / 2, drawer_half_width, panel / 2],
        pose=[-drawer_depth / 2, 0, drawer_floor_z],
        material=green_dark,
        density=500.0,
    )
    for side in (-1.0, 1.0):
        _add_box(
            moving,
            half_size=[drawer_depth / 2, panel / 2, height * 0.34],
            pose=[-drawer_depth / 2, side * drawer_half_width, height * 0.36],
            material=green_dark,
            density=500.0,
        )

    # A fixed child link gives the handle its own segmentation ID while
    # preserving rigid attachment to the moving drawer body.
    handle = builder.create_link_builder(moving)
    handle.set_name("drawer_object_handle")
    handle.set_joint_name("drawer_handle_fixed_joint")
    handle.set_joint_properties(
        type="fixed",
        limits=[],
        pose_in_parent=sapien.Pose(),
        pose_in_child=sapien.Pose(),
    )
    bar_x = front_x + handle_standoff
    _add_box(
        handle,
        half_size=[handle_radius, handle_width / 2, handle_radius],
        pose=[bar_x, 0, handle_z],
        material=black,
        density=900.0,
    )
    support_half = max(handle_radius, (handle_standoff - handle_radius) / 2)
    for side in (-1.0, 1.0):
        _add_box(
            handle,
            half_size=[support_half, handle_radius, handle_radius],
            pose=[front_x + support_half, side * handle_width / 2, handle_z],
            material=black,
            density=900.0,
        )

    articulation = builder.build("lfv_drawer", fix_root_link=True)
    return articulation


@register_env("LFVOpenDrawer-v1", max_episode_steps=400)
class LFVOpenDrawerEnv(LFVTabletopBaseEnv):
    """Dataset-aligned tabletop drawer for affordance-to-motion validation."""

    # Dataset/pouring-aligned table-front view: camera on world +X, Panda on
    # world -X, and yaw=0 makes the drawer handle face the camera.  The RGB-D
    # inference camera deliberately matches the pouring baseline convention.
    sensor_camera_eye = [0.50, 0.0, 0.52]
    sensor_camera_target = [-0.06, 0.0, 0.035]
    opposite_camera_eye = [-0.55, 0.30, 0.58]
    opposite_camera_target = [-0.06, 0.0, 0.045]
    record_camera_eye = [0.68, -0.48, 0.58]
    record_camera_target = [-0.08, 0.0, 0.055]
    render_camera_eye = [0.58, -0.38, 0.54]
    render_camera_target = [-0.08, 0.0, 0.055]

    def __init__(self, *args, drawer_geometry: dict | None = None, **kwargs):
        self.drawer_geometry = dict(drawer_geometry or {})
        self.open_limit = float(self.drawer_geometry.get("open_limit", 0.19))
        # The accepted demonstrations have about 9.3 cm median 3-D endpoint
        # motion and about 6.8 cm mean motion along this scene's joint axis.
        self.success_threshold = float(
            self.drawer_geometry.get("success_threshold", 0.060)
        )
        super().__init__(*args, **kwargs)

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        self.drawer = build_lfv_drawer(self.scene, self.drawer_geometry)
        self.cabinet_link = self.drawer.links_map["drawer_object_cabinet"]
        self.moving_link = self.drawer.links_map["drawer_object_moving_body"]
        self.handle_link = self.drawer.links_map["drawer_object_handle"]

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            options = options or {}
            layout = options.get("layout", {})
            batch_size = len(env_idx)
            self.table_scene.initialize(env_idx)
            self._initialize_robot()

            drawer_xy = layout.get("drawer_xy", [-0.08, 0.02])
            drawer_yaw = float(layout.get("drawer_yaw", 0.0))
            drawer_z = float(layout.get("drawer_z", 0.004))
            p = torch.tensor(
                [[float(drawer_xy[0]), float(drawer_xy[1]), drawer_z]],
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            q = torch.tensor(
                euler2quat(0.0, 0.0, drawer_yaw, axes="sxyz"),
                dtype=torch.float32,
                device=self.device,
            ).repeat(batch_size, 1)
            self.drawer.set_pose(Pose.create_from_pq(p=p, q=q))
            initial_open = float(layout.get("initial_open", 0.0))
            qpos = torch.full(
                (batch_size, 1),
                min(max(initial_open, 0.0), self.open_limit),
                dtype=torch.float32,
                device=self.device,
            )
            self.drawer.set_qpos(qpos)
            self.drawer.set_qvel(torch.zeros_like(qpos))

    @property
    def drawer_qpos(self):
        return self.drawer.get_qpos()[:, 0]

    def evaluate(self):
        qpos = self.drawer_qpos
        grasped = self.agent.is_grasping(self.handle_link)
        return {
            "success": qpos >= self.success_threshold,
            "drawer_qpos": qpos,
            "open_fraction": qpos / self.open_limit,
            "is_grasped": grasped,
        }

    def _get_obs_extra(self, info: dict):
        obs = {
            "tcp_pose": self.agent.tcp.pose.raw_pose,
            "is_grasped": info["is_grasped"],
            "drawer_qpos": info["drawer_qpos"],
            "handle_pose": self.handle_link.pose.raw_pose,
        }
        if "state" in self.obs_mode:
            obs.update(
                cabinet_pose=self.cabinet_link.pose.raw_pose,
                moving_pose=self.moving_link.pose.raw_pose,
            )
        return obs
