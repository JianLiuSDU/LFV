from __future__ import annotations

from .specs import ManiSkillTaskSpec


def make_env(
    spec: ManiSkillTaskSpec,
    *,
    robot_uids: str | None = None,
    obs_mode: str | None = None,
    control_mode: str | None = None,
    render_mode: str = "rgb_array",
    shader: str = "default",
    num_envs: int = 1,
    extra_env_kwargs: dict | None = None,
):
    try:
        import gymnasium as gym
        import mani_skill.envs  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ManiSkill is unavailable. Run this stage in the maniskill3 conda environment."
        ) from exc
    # Keep project registration errors visible instead of incorrectly reporting
    # them as a missing third-party ManiSkill installation.
    import lfv_sim.maniskill.envs  # noqa: F401

    sensor_configs = {
        spec.camera_uid: {
            "width": spec.sensor_width,
            "height": spec.sensor_height,
            "shader_pack": shader,
        }
    }
    render_configs = {
        "render_camera": {
            "width": spec.render_width,
            "height": spec.render_height,
            "shader_pack": shader,
        }
    }
    kwargs = dict(spec.env_kwargs)
    if extra_env_kwargs:
        kwargs.update(extra_env_kwargs)
    kwargs.update(
        obs_mode=obs_mode or spec.obs_mode,
        control_mode=control_mode or spec.control_mode,
        render_mode=render_mode,
        robot_uids=robot_uids or spec.robot_uids,
        num_envs=num_envs,
        sensor_configs=sensor_configs,
        human_render_camera_configs=render_configs,
    )
    return gym.make(spec.env_id, **kwargs)
