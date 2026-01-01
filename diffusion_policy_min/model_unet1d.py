# diffusion_policy_min/model_unet1d.py
import torch
import torch.nn as nn
from diffusers import UNet1DModel

class DiffusionPolicy1D(nn.Module):
    """
    Denoises action trajectories conditioned on an observation window.
    """
    def __init__(self, obs_dim=31, act_dim=8, obs_horizon=2, act_horizon=16):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.obs_h = obs_horizon
        self.act_h = act_horizon

        cond_dim = obs_dim * obs_horizon
        in_channels = act_dim + cond_dim
        out_channels = act_dim

        # UNet1DModel expects x: [B, C, L]
        self.unet = UNet1DModel(
            sample_size=act_horizon,     # length L
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 256, 256),
            down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
        )

    def forward(self, noisy_actions, timesteps, obs_seq):
        """
        noisy_actions: [B, L, act_dim]
        obs_seq:       [B, obs_h, obs_dim]
        timesteps:     [B] or scalar tensor
        returns: predicted noise [B, L, act_dim]
        """
        B, L, A = noisy_actions.shape

        # cond: [B, obs_h*obs_dim]
        cond = obs_seq.reshape(B, -1)

        # repeat cond across time => [B, L, cond_dim]
        cond_rep = cond[:, None, :].repeat(1, L, 1)

        # concat => [B, L, act_dim + cond_dim]
        x = torch.cat([noisy_actions, cond_rep], dim=-1)

        # to [B, C, L]
        x = x.permute(0, 2, 1)

        out = self.unet(x, timesteps).sample  # [B, act_dim, L]
        out = out.permute(0, 2, 1)            # [B, L, act_dim]
        return out