import numpy as np
import zarr
import time
import os

# Import the class you defined in Step 1
from env.sim_env import StackingEnv
# Import the expert planner
from expert.motion_planner import StackingExpert

def main():
    # 1. Initialize Headless Environment (gui=False for server)
    env = StackingEnv(gui=False)
    
    # 2. Initialize Expert
    expert = StackingExpert(env)
    
    # 3. Setup Output File
    output_path = 'data/stacking_demo.zarr'
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    
    # mode='w' overwrites previous data so you get a clean start
    root = zarr.open_group(output_path, mode='w')
    
    print(f"--- Starting Data Collection for {output_path} ---")
    
    # Lists to store data
    all_obs = []
    all_actions = []
    episode_ends = []
    
    num_episodes = 5 
    
    for ep in range(num_episodes):
        print(f"Episode {ep+1}/{num_episodes} Start...")
        
        # Reset env (this triggers your safe random spawning)
        env.reset()
        
        # Expert plans the trajectory
        expert_actions = expert.generate_episode()
        
        if len(expert_actions) == 0:
            print("⚠️ Warning: Expert generated 0 steps. Skipping.")
            continue

        # Execute the plan in the environment
        for action in expert_actions:
            obs = env.step(action)
            
            # Flatten Observation for Diffusion Policy
            # Obs = [Robot Joints (7)] + [Cube 1 Pos/Orn (7)] + [Cube 2...] + [Cube 3...]
            # Total Dim = 7 + 21 = 28
            flat_cubes = np.array(obs['cubes']).flatten()
            flat_obs = np.concatenate([obs['robot'], flat_cubes])
            
            all_obs.append(flat_obs)
            all_actions.append(action)

        # Mark the end of this episode
        episode_ends.append(len(all_actions))
        
        if (ep+1) % 10 == 0:
            print(f"✅ Completed {ep+1} episodes.")

    # 4. Save Arrays to Zarr
    print("Saving to disk...")
    root.create_group('data')
    root.create_group('meta')
    
    # Compress and save
    root['data/action'] = np.array(all_actions, dtype=np.float32)
    root['data/state'] = np.array(all_obs, dtype=np.float32)
    root['meta/episode_ends'] = np.array(episode_ends, dtype=np.int32)
    
    print(f"🎉 Success! Total steps collected: {len(all_actions)}")

# --- ENTRY POINT (Crucial!) ---
if __name__ == "__main__":
    main()