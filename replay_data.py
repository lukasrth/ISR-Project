import zarr
import pybullet as p
import pybullet_data
import numpy as np
import cv2

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

# --- CRITICAL DEBUG: CHECK DATA BEFORE RENDERING ---
print(f"\n🔍 DEBUGGING FRAME {start_idx} (First frame of replay)")
first_frame = states[start_idx]
# Robot: 0-7
# Cube 0: 7-14
# Cube 1: 14-21
# Cube 2: 21-28
c0_pos = first_frame[7:10]
c1_pos = first_frame[14:17]
c2_pos = first_frame[21:24]

print(f"   🔴 Cube 0 (Red)   Pos: {np.round(c0_pos, 3)}")
print(f"   🟢 Cube 1 (Green) Pos: {np.round(c1_pos, 3)}")
print(f"   🔵 Cube 2 (Blue)  Pos: {np.round(c2_pos, 3)}")

if np.linalg.norm(c1_pos) < 0.01:
    print("   ⚠️ WARNING: Cube 1 is at (0,0,0). The DATA generation might be failing.")
else:
    print("   ✅ Coordinates look valid. Proceeding to render.")
# ---------------------------------------------------

# 2. Setup Sim
p.connect(p.DIRECT) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

p.loadURDF("plane.urdf")
p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)

# 3. Setup Ghosts
cube_visuals = []
colors = [[1,0,0,1], [0,1,0,1], [0,0,1,1]]
sizes = [0.05, 0.04, 0.03]

for i in range(3):
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[sizes[i]/2]*3, rgbaColor=colors[i])
    body = p.createMultiBody(baseVisualShapeIndex=vis)
    cube_visuals.append(body)

# 4. Video Setup
width, height = 640, 480
view_matrix = p.computeViewMatrix(
    cameraEyePosition=[1.0, -0.5, 0.8], # High angle side view
    cameraTargetPosition=[0.5, 0, 0.0],   
    cameraUpVector=[0, 0, 1]
)
proj_matrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100)
video_writer = cv2.VideoWriter('replay.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

print(f"\n🎥 Rendering {end_idx - start_idx} frames to replay.mp4...")

for i in range(start_idx, end_idx):
    current_state = states[i]
    
    # Update Robot
    for j in range(7):
        p.resetJointState(robot, j, current_state[j])
        
    # Update Cubes (FIXED INDEXING)
    # Stride is 7 (3 pos + 4 orn)
    for c_idx in range(3):
        idx_start = 7 + (c_idx * 7)
        pos = current_state[idx_start : idx_start+3]
        orn = current_state[idx_start+3 : idx_start+7]
        p.resetBasePositionAndOrientation(cube_visuals[c_idx], pos, orn)

    # Gripper
    gripper_val = actions[i][-1] * 0.04
    p.resetJointState(robot, 9, gripper_val)
    p.resetJointState(robot, 10, gripper_val)

    # Capture
    _, _, px, _, _ = p.getCameraImage(width, height, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
    img = np.reshape(np.array(px, dtype=np.uint8), (height, width, 4))[:, :, :3]
    video_writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

video_writer.release()
p.disconnect()
print("✅ Done.")