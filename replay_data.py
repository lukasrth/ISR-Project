import zarr
import pybullet as p
import pybullet_data
import numpy as np
import cv2

# --- CONFIG ---
# Map sizes to the colors used in SimEnv
# 0.05 -> Red, 0.04 -> Green, 0.03 -> Blue
SIZE_TO_COLOR = {
    0.05: [1, 0, 0, 1],
    0.04: [0, 1, 0, 1],
    0.03: [0, 0, 1, 1]
}

def get_color_for_size(size_val):
    # Find closest key in case of tiny floating point differences
    # e.g. 0.0499999 -> 0.05
    closest_size = min(SIZE_TO_COLOR.keys(), key=lambda x: abs(x - size_val))
    return SIZE_TO_COLOR[closest_size]

# 1. Load Data
try:
    dataset = zarr.open('data/stacking_demo.zarr', mode='r')
    states = dataset['data/state'][:]
    actions = dataset['data/action'][:]
    episode_ends = dataset['meta/episode_ends'][:]
except Exception as e:
    print(f"❌ Error loading file: {e}")
    exit()

# Select LAST episode
if len(episode_ends) == 0:
    print("Dataset is empty.")
    exit()

start_idx = 0 if len(episode_ends) == 1 else episode_ends[-2]
end_idx = episode_ends[-1]
first_frame = states[start_idx]

# 2. Setup Sim
p.connect(p.DIRECT) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# 3. Dynamic Cube Setup
cube_visuals = []

print(f"\n🔍 Analyzing Episode Structure (Start Frame: {start_idx})")

# We loop 3 times (for 3 cubes)
for i in range(3):
    # Calculate index in the state vector
    # 7 (Robot) + i * 8 (Stride) -> 8 is the new size (7 pos/orn + 1 size)
    base_idx = 7 + (i * 8)
    
    # Read Position (Indices 0-3 of the chunk)
    pos = first_frame[base_idx : base_idx+3]
    
    # Read Size (Index 7 of the chunk)
    size = first_frame[base_idx + 7]
    
    color = get_color_for_size(size)
    print(f"   Cube {i}: Size={size:.3f} -> Color={color}")

    # Create Visual
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=color)
    body = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=pos)
    cube_visuals.append(body)

# 4. Video Setup
width, height = 640, 480
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1.0, -0.5, 0.8], 
    cameraTargetPosition=[0.5, 0, 0.0],   
    cameraUpVector=[0, 0, 1]
)
proj_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100)
video_writer = cv2.VideoWriter('replay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

print(f"🎥 Rendering {end_idx - start_idx} frames...")

for i in range(start_idx, end_idx):
    current_state = states[i]
    
    # Update Robot
    for j in range(7):
        p.resetJointState(robot, j, current_state[j])
        
    # Update Cubes
    for c_idx in range(3):
        # STRIDE IS 8 (Pos[3] + Orn[4] + Size[1])
        idx_start = 7 + (c_idx * 8)
        
        pos = current_state[idx_start : idx_start+3]
        orn = current_state[idx_start+3 : idx_start+7]
        # We ignore index+7 (Size) here because size doesn't change mid-video
        
        p.resetBasePositionAndOrientation(cube_visuals[c_idx], pos, orn)

    # Update Gripper
    gripper_val = actions[i][-1] * 0.04
    p.resetJointState(robot, 9, gripper_val)
    p.resetJointState(robot, 10, gripper_val)

    # Capture
    _, _, px, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    img = np.reshape(np.array(px, dtype=np.uint8), (height, width, 4))[:, :, :3]
    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

video_writer.release()
p.disconnect()
print("✅ Done. Check replay.mp4")