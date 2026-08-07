"""Diffusers scheduler factories with one shared configuration."""

from __future__ import annotations

from diffusers import DDIMScheduler, DDPMScheduler


def scheduler_kwargs(
    num_train_timesteps: int = 100,
    beta_schedule: str = "squaredcos_cap_v2",
    prediction_type: str = "sample",
) -> dict:
    return {
        "num_train_timesteps": int(num_train_timesteps),
        "beta_schedule": str(beta_schedule),
        "prediction_type": str(prediction_type),
        "clip_sample": False,
    }


def make_ddpm_scheduler(**kwargs) -> DDPMScheduler:
    return DDPMScheduler(**scheduler_kwargs(**kwargs))


def make_ddim_scheduler(**kwargs) -> DDIMScheduler:
    return DDIMScheduler(
        **scheduler_kwargs(**kwargs),
        set_alpha_to_one=True,
    )
