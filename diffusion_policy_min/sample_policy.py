# diffusion_policy_min/sample_policy.py
import torch
from diffusers import DDPMScheduler
from model_unet1d import DiffusionPolicy1D

@torch.no_grad()
def sample_actions(model, obs_seq, num_steps=50, device="cpu"):
    """
    obs_seq: [B, obs_h, obs_dim] (already normalized)
    returns: [B, L, act_dim] normalized
    """
    model.eval()
    B, obs_h, obs_dim = obs_seq.shape
    L = model.act_h
    act_dim = model.act_dim

    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )
    scheduler.set_timesteps(num_steps, device=device)

    x = torch.randn((B, L, act_dim), device=device)

    for t in scheduler.timesteps:
        # model predicts epsilon
        eps = model(x, t, obs_seq)
        step_out = scheduler.step(eps, t, x)
        x = step_out.prev_sample

    return x