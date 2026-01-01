import torch
import numpy as np

from model_unet1d import DiffusionPolicy1D
from sample_policy import sample_actions

# ---- import YOUR environment ----
# adjust this to match your repo
from env.sim_env import StackingEnv


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # -------- load checkpoint --------
    ckpt_path = "checkpoints/dp_state_only_step20000.pt"
    ckpt = torch.load(ckpt_path, map_location=device)

    obs_h = ckpt["obs_horizon"]
    act_h = ckpt["act_horizon"]

    state_mean = ckpt["state_mean"]
    state_std  = ckpt["state_std"]
    act_mean   = ckpt["act_mean"]
    act_std    = ckpt["act_std"]

    model = DiffusionPolicy1D(
        obs_dim=31,
        act_dim=8,
        obs_horizon=obs_h,
        act_horizon=act_h,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    # -------- environment --------
    env = StackingEnv(gui=True)   # set gui=False for speed

    obs = env.reset()             # shape: [31]
    obs_hist = [obs.copy() for _ in range(obs_h)]

    done = False
    step = 0

    while not done:
        # ---- build observation window ----
        obs_seq = np.stack(obs_hist[-obs_h:], axis=0)
        obs_seq = (obs_seq - state_mean) / state_std
        obs_seq_t = torch.tensor(
            obs_seq, dtype=torch.float32, device=device
        )[None, ...]

        # ---- sample action trajectory ----
        act_seq_norm = sample_actions(
            model, obs_seq_t, num_steps=50, device=device
        )[0].cpu().numpy()

        # ---- denormalize & execute first action ----
        act_seq = act_seq_norm * act_std + act_mean
        action = act_seq[0]

        obs, reward, done, info = env.step(action)
        obs_hist.append(obs)

        step += 1
        print(f"step {step}, reward {reward}")

    env.close()


if __name__ == "__main__":
    main()