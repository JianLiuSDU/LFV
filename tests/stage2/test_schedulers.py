import torch

from lfv.diffusion import make_ddim_scheduler, make_ddpm_scheduler


def test_add_noise_and_sample_prediction_step():
    clean = torch.randn(2, 9)
    noise = torch.randn_like(clean)
    timesteps = torch.tensor([5, 9], dtype=torch.long)
    ddpm = make_ddpm_scheduler(num_train_timesteps=20)
    noisy = ddpm.add_noise(clean, noise, timesteps)
    assert noisy.shape == clean.shape
    assert torch.isfinite(noisy).all()
    ddim = make_ddim_scheduler(num_train_timesteps=20)
    ddim.set_timesteps(5)
    state = torch.randn_like(clean)
    for timestep in ddim.timesteps:
        state = ddim.step(clean, timestep, state, eta=0.0).prev_sample
    assert torch.isfinite(state).all()
