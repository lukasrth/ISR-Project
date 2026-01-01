# diffusion_policy_min/dataset_zarr.py
import numpy as np
import zarr

class ZarrSequenceDataset:
    """
    Samples (obs_seq, act_seq) from a zarr dataset with episode boundaries.
    obs_seq: [obs_horizon, obs_dim]
    act_seq: [act_horizon, act_dim]
    """
    def __init__(self, zarr_path, obs_horizon=2, act_horizon=16, seed=0):
        self.z = zarr.open(zarr_path, mode="r")
        self.states = np.array(self.z["data/state"])   # [N, 31]
        self.actions = np.array(self.z["data/action"]) # [N, 8]
        self.ep_ends = np.array(self.z["meta/episode_ends"])  # [num_episodes]
        self.obs_h = int(obs_horizon)
        self.act_h = int(act_horizon)

        assert self.states.shape[0] == self.actions.shape[0]
        self.N = self.states.shape[0]

        # Build (episode_start, episode_end) pairs
        self.episodes = []
        start = 0
        for end in self.ep_ends:
            self.episodes.append((start, int(end)))
            start = int(end)

        self.rng = np.random.default_rng(seed)

        # Precompute valid starting indices inside each episode
        # We need enough room for obs_horizon (past including t) and act_horizon (future from t)
        # We'll define "t" as the last index of the obs window.
        self.valid = []
        for (s, e) in self.episodes:
            # t must satisfy:
            # obs window uses [t-obs_h+1 ... t]  => t >= s + obs_h - 1
            # future actions use [t ... t+act_h-1] => t+act_h-1 < e
            t_min = s + self.obs_h - 1
            t_max = e - self.act_h
            if t_min <= t_max:
                for t in range(t_min, t_max + 1):
                    self.valid.append(t)
        self.valid = np.array(self.valid, dtype=np.int64)
        if len(self.valid) == 0:
            raise ValueError("No valid sequences. Try reducing obs_horizon / act_horizon.")

        # Normalize (important for stable diffusion)
        self.state_mean = self.states.mean(axis=0)
        self.state_std  = self.states.std(axis=0) + 1e-6
        self.act_mean   = self.actions.mean(axis=0)
        self.act_std    = self.actions.std(axis=0) + 1e-6

    def __len__(self):
        return len(self.valid)

    def sample_batch(self, batch_size):
        idx = self.rng.integers(0, len(self.valid), size=batch_size)
        t = self.valid[idx]  # [B]

        obs_seq = np.stack([self.states[ti - self.obs_h + 1: ti + 1] for ti in t], axis=0)
        act_seq = np.stack([self.actions[ti: ti + self.act_h] for ti in t], axis=0)

        # normalize
        obs_seq = (obs_seq - self.state_mean) / self.state_std
        act_seq = (act_seq - self.act_mean) / self.act_std

        return obs_seq.astype(np.float32), act_seq.astype(np.float32)

    def denorm_action(self, act_seq_norm):
        return act_seq_norm * self.act_std + self.act_mean