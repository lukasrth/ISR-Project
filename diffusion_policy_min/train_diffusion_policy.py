# diffusion_policy_min/train_diffusion_policy.py
import os
import math
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from diffusers import DDPMScheduler

from dataset_zarr import ZarrSequenceDataset
from model_unet1d import DiffusionPolicy1D

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Device:", device)

    # ====== config ======
    zarr_path = "../data/stacking_demo.zarr"
    obs_horizon = 2
    act_horizon = 16
    batch_size = 64
    steps = 30_000
    lr = 1e-4
    save_every = 2000
    outdir = "checkpoints"
    os.makedirs(outdir, exist_ok=True)

    # ====== data ======
    ds = ZarrSequenceDataset(zarr_path, obs_horizon=obs_horizon, act_horizon=act_horizon, seed=0)

    obs_dim = ds.states.shape[1]
    act_dim = ds.actions.shape[1]

    # ====== model + diffusion ======
    model = DiffusionPolicy1D(obs_dim=obs_dim, act_dim=act_dim,
                              obs_horizon=obs_horizon, act_horizon=act_horizon).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon"
    )

    opt = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ====== training loop ======
    model.train()
    for step in range(1, steps + 1):
        obs_seq_np, act_seq_np = ds.sample_batch(batch_size)

        obs_seq = torch.from_numpy(obs_seq_np).to(device)   # [B, obs_h, obs_dim]
        act_seq = torch.from_numpy(act_seq_np).to(device)   # [B, L, act_dim]

        # sample timestep per batch element
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (batch_size,),
            device=device, dtype=torch.long
        )

        noise = torch.randn_like(act_seq)
        noisy_actions = noise_scheduler.add_noise(act_seq, noise, timesteps)

        pred_noise = model(noisy_actions, timesteps, obs_seq)
        loss = F.mse_loss(pred_noise, noise)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 100 == 0:
            print(f"step {step:06d} | loss {loss.item():.6f}")

        if step % save_every == 0:
            ckpt = {
                "model": model.state_dict(),
                "obs_horizon": obs_horizon,
                "act_horizon": act_horizon,
                "state_mean": ds.state_mean,
                "state_std": ds.state_std,
                "act_mean": ds.act_mean,
                "act_std": ds.act_std,
            }
            path = os.path.join(outdir, f"dp_state_only_step{step}.pt")
            torch.save(ckpt, path)
            print("saved:", path)

    print("done.")

if __name__ == "__main__":
    main()