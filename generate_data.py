import numpy as np
import zarr
import time
import os
import argparse
from env.sim_env import StackingEnv
from expert.motion_planner import StackingExpert

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Generate demonstration data for Stacking Task")
    
    # Flag: Save Images? (True/False)
    parser.add_argument('--visual', action='store_true', help="Save camera images for visual policy training")
    
    # Flag: Number of Episodes (Int)
    parser.add_argument('--num_episodes', type=int, default=50, help="Number of episodes to generate (Default: 50)")
    
    args = parser.parse_args()
    
    SAVE_VISUAL = args.visual
    NUM_EPISODES = args.num_episodes
    
    # 2. Init Env
    env = StackingEnv(gui=False)
    expert = StackingExpert(env)
    
    # 3. Setup Output
    os.makedirs('data', exist_ok=True)
    output_path = 'data/stacking_demo.zarr'
    root = zarr.open_group(output_path, mode='w')
    
    print(f"--- Starting Data Collection ---")
    print(f"    Target File: {output_path}")
    print(f"    Episodes:    {NUM_EPISODES}")
    print(f"    Visual Mode: {'ON 📸' if SAVE_VISUAL else 'OFF ❌ (State only)'}")
    
    # Storage
    all_obs = []
    all_actions = []
    all_images = [] 
    episode_ends = []
    
    for ep in range(NUM_EPISODES):
        print(f"Episode {ep+1}/{NUM_EPISODES} Start...")
        
        env.reset()
        expert_actions = expert.generate_episode()
        
        if len(expert_actions) == 0:
            print("⚠️ Warning: Expert generated 0 steps.")
            continue

        for action in expert_actions:
            obs = env.step(action)
            
            # 1. Save State Data (Always)
            flat_cubes = np.array(obs['cubes']).flatten()
            flat_obs = np.concatenate([obs['robot'], flat_cubes])
            all_obs.append(flat_obs)
            all_actions.append(action)
            
            # 2. Save Visual Data (Conditional)
            if SAVE_VISUAL:
                img = env.get_image()
                all_images.append(img)

        episode_ends.append(len(all_actions))
        
        if (ep+1) % 10 == 0:
            print(f"✅ Completed {ep+1} episodes.")

    # 4. Save to Zarr
    print("Saving to disk...")
    root.create_group('data')
    root.create_group('meta')
    
    # Save State & Actions
    root['data/action'] = np.array(all_actions, dtype=np.float32)
    root['data/state'] = np.array(all_obs, dtype=np.float32)
    root['meta/episode_ends'] = np.array(episode_ends, dtype=np.int32)
    
    # Save Images (If enabled)
    if SAVE_VISUAL:
        print(f"    Compressing {len(all_images)} images (this may take a moment)...")
        root.create_dataset(
            'data/img', 
            data=np.array(all_images, dtype=np.uint8), 
            chunks=(100, 96, 96, 3),
            compressor=zarr.Blosc(cname='zstd', clevel=3, shuffle=2)
        )
        print("    Images saved to data/img")
    
    print(f"🎉 Success! Total steps: {len(all_actions)}")

if __name__ == "__main__":
    main()