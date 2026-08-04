"""Panda with wide, long, high-friction parallel finger extensions."""

from __future__ import annotations

import sapien

from mani_skill import format_path
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots import Panda
from mani_skill.utils import sapien_utils

from lfv.robot.gripper_extension import (
    DEFAULT_LONG_FINGER_SPEC,
    DRAWER_LONG_FINGER_SPEC,
)


@register_agent()
class PandaLongFinger(Panda):
    """Stock Panda kinematics plus UMI/Fin-Ray-inspired rigid finger plates.

    Only the finger-link visual and collision geometry changes.  Joint names,
    opening limits, controller dimensions and ``panda_hand_tcp`` are identical
    to the stock Panda, which keeps existing LFV grasp and motion interfaces
    valid.  The rigid plate is an intentionally simple simulation proxy; its
    high-friction material approximates a compliant TPU/rubber contact skin.
    """

    uid = "panda_long_finger"
    extension_spec = DEFAULT_LONG_FINGER_SPEC

    def _load_articulation(self, initial_pose=None):
        if self.build_separate:
            raise NotImplementedError(
                "panda_long_finger currently supports a shared articulation only"
            )

        loader = self.scene.create_urdf_loader()
        asset_path = format_path(str(self.urdf_path))
        loader.name = self.uid
        if self._agent_idx is not None:
            loader.name = f"{self.uid}-agent-{self._agent_idx}"
        loader.fix_root_link = self.fix_root_link
        loader.load_multiple_collisions_from_file = self.load_multiple_collisions
        loader.disable_self_collisions = self.disable_self_collisions

        if self.urdf_config is not None:
            urdf_config = sapien_utils.parse_urdf_config(self.urdf_config)
            sapien_utils.check_urdf_config(urdf_config)
            sapien_utils.apply_urdf_config(loader, urdf_config)

        builder = loader.parse(asset_path)["articulation_builders"][0]
        builder.initial_pose = initial_pose
        visual_material = sapien.render.RenderMaterial(
            base_color=[0.055, 0.065, 0.075, 1.0],
            roughness=0.88,
            metallic=0.0,
        )
        pad_material = sapien.render.RenderMaterial(
            base_color=[0.93, 0.34, 0.08, 1.0],
            roughness=0.95,
            metallic=0.0,
        )
        contact_material = sapien.physx.PhysxMaterial(
            self.extension_spec.static_friction,
            self.extension_spec.dynamic_friction,
            0.0,
        )

        links = {link.name: link for link in builder.link_builders}
        for side, link_name in (
            ("left", "panda_leftfinger"),
            ("right", "panda_rightfinger"),
        ):
            link = links[link_name]
            center = self.extension_spec.center_for_side(side)
            pose = sapien.Pose(p=center)
            link.add_box_visual(
                pose=pose,
                half_size=self.extension_spec.half_size_m,
                material=visual_material,
                name=f"{side}_long_finger_body",
            )
            link.add_box_collision(
                pose=pose,
                half_size=self.extension_spec.half_size_m,
                material=contact_material,
                density=self.extension_spec.density_kg_m3,
                patch_radius=0.01,
                min_patch_radius=0.005,
            )

            # A thin orange strip makes the enlarged inner contact face easy
            # to inspect in the saved front/oblique videos.
            inner_y = 0.0003 if side == "left" else -0.0003
            link.add_box_visual(
                pose=sapien.Pose(p=[0.0, inner_y, self.extension_spec.center_z_m]),
                half_size=[
                    self.extension_spec.contact_width_m / 2 * 0.94,
                    0.00035,
                    self.extension_spec.contact_length_m / 2 * 0.96,
                ],
                material=pad_material,
                name=f"{side}_long_finger_contact_skin",
            )

        self.robot = builder.build()
        if self.robot is None:
            raise RuntimeError(f"Failed to build custom robot from {asset_path}")
        self.robot_link_names = [link.name for link in self.robot.get_links()]


@register_agent()
class PandaDrawerFinger(PandaLongFinger):
    """Long high-friction plates narrowed for a tabletop drawer handle."""

    uid = "panda_drawer_finger"
    extension_spec = DRAWER_LONG_FINGER_SPEC
