from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class ManiSkillTaskSpec:
    name: str
    env_id: str
    robot_uids: str
    instruction: str
    manipulated_query: str
    target_query: Optional[str]
    object_query: Optional[str] = None
    manipulated_entity_attr: Optional[str] = None
    reference_entity_attr: Optional[str] = None
    camera_uid: str = "base_camera"
    obs_mode: str = "rgb+depth+segmentation"
    control_mode: str = "pd_ee_delta_pose"
    sensor_width: int = 640
    sensor_height: int = 480
    render_width: int = 1280
    render_height: int = 720
    env_kwargs: dict = field(default_factory=dict)


TASK_SPECS = {
    "pouring": ManiSkillTaskSpec(
        name="pouring",
        env_id="LFVPourCupBowl-v1",
        robot_uids="panda",
        instruction="Pour water from the cup into the bowl",
        manipulated_query="cup",
        target_query="bowl",
        object_query="cup",
        manipulated_entity_attr="cup",
        reference_entity_attr="bowl",
    ),
    "picknplace": ManiSkillTaskSpec(
        name="picknplace",
        env_id="LFVPickBananaPlate-v1",
        robot_uids="panda",
        instruction="Pick up the banana and place it on the plate",
        manipulated_query="banana",
        target_query="plate",
        object_query="banana",
        manipulated_entity_attr="banana",
        reference_entity_attr="plate",
    ),
    "drawer_open": ManiSkillTaskSpec(
        name="drawer_open",
        env_id="LFVOpenDrawer-v1",
        robot_uids="panda_drawer_finger",
        instruction="Pull the black drawer handle to open the drawer",
        manipulated_query="drawer_object_handle",
        target_query="drawer_object_cabinet",
        object_query="drawer_object",
        manipulated_entity_attr="handle_link",
        reference_entity_attr="cabinet_link",
        env_kwargs={
            "drawer_geometry": {
                "width": 0.24,
                "depth": 0.26,
                # Match the shallow tabletop drawer used in both drawer
                # datasets; top-down reachability is handled by grasp/TCP
                # calibration rather than changing the object's scale.
                "height": 0.095,
                "depth": 0.28,
                "open_limit": 0.19,
                "success_threshold": 0.060,
                "handle_width": 0.100,
                "handle_radius": 0.008,
                "handle_standoff": 0.032,
                "handle_z": 0.042,
            }
        },
    ),
}


def get_task_spec(task: str) -> ManiSkillTaskSpec:
    try:
        return TASK_SPECS[task]
    except KeyError as exc:
        raise KeyError(f"Unknown LFV ManiSkill task {task!r}; known={sorted(TASK_SPECS)}") from exc
