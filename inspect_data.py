import zarr
import numpy as np

def inspect():
    path = "data/stacking_demo.zarr"
    print(f"Loading {path}...")
    
    root = zarr.open(path, mode='r')
    
    # 1. check shapes
    actions = root['data/action'][:]
    states = root['data/state'][:]
    ends = root['meta/episode_ends'][:]
    
    print("\n--- Dimensions ---")
    print(f"Total Steps Recorded: {actions.shape[0]}")
    print(f"Action Dimension:     {actions.shape[1]} (Should be 8: 7 joints + 1 gripper)")
    print(f"State Dimension:      {states.shape[1]} (Should be ~16: 7 joints + 9 cube coords)")
    print(f"Total Episodes:       {len(ends)}")
    
    # 2. Check Data Integrity
    print("\n--- Values ---")
    print(f"Action Range: Min={np.min(actions):.3f}, Max={np.max(actions):.3f}")
    print(f"State Range:  Min={np.min(states):.3f},  Max={np.max(states):.3f}")
    
    if np.isnan(actions).any() or np.isnan(states).any():
        print("\n❌ CRITICAL ERROR: Data contains NaNs!")
    else:
        print("\n✅ Data looks clean (No NaNs).")

if __name__ == "__main__":
    inspect()