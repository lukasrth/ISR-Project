import pybullet as p
import pybullet_data
import numpy as np
import time
import random

class StackingEnv:
    def __init__(self, gui=False):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        self.dt = 1./240.
        
        self.plane_id = p.loadURDF("plane.urdf")
        self.table_id = p.loadURDF("table/table.urdf", basePosition=[0.5, 0, -0.63])
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True)
        
        # We define the available types here
        # (Size, RGBA Color)
        self.cube_types = [
            (0.05, [1, 0, 0, 1]), # Large = Red
            (0.04, [0, 1, 0, 1]), # Med   = Green
            (0.03, [0, 0, 1, 1])  # Small = Blue
        ]
        
        self.cube_ids = []
        self.id_to_size = {} 

        # Robot Dynamics
        for link_id in [9, 10]:
            p.changeDynamics(
                self.robot_id, link_id, 
                lateralFriction=1.0, 
                spinningFriction=0.1, 
                ccdSweptSphereRadius=0.002, 
                contactStiffness=10000.0, 
                contactDamping=1.0 
            )

    def get_image(self):
        view_matrix = p.computeViewMatrix([1.0, 0.0, 0.5], [0.5, 0.0, 0.1], [0, 0, 1])
        proj_matrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
        _, _, px, _, _ = p.getCameraImage(96, 96, view_matrix, proj_matrix, renderer=p.ER_TINY_RENDERER)
        rgb_array = np.array(px, dtype=np.uint8).reshape((96, 96, 4))[:, :, :3]
        return rgb_array

    def reset(self):
        # Reset Robot
        for i, val in enumerate([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4, 0.04, 0.04]):
            p.resetJointState(self.robot_id, i, val)

        # Clear Cubes
        for uid in self.cube_ids: p.removeBody(uid)
        self.cube_ids = []
        self.id_to_size = {}

        # --- RANDOMIZE SEQUENCE ---
        # We copy the list and shuffle it so the order (0, 1, 2) is different every time.
        current_episode_cubes = self.cube_types.copy()
        random.shuffle(current_episode_cubes)

        spawn_x = [0.35, 0.65]
        spawn_y = [0.0, 0.25]

        for size, color in current_episode_cubes:
            pos = [0,0,0]
            valid = False
            
            # Find valid spawn position
            for _ in range(100):
                x, y = np.random.uniform(*spawn_x), np.random.uniform(*spawn_y)
                # Check distance against all currently spawned cubes
                if not any(np.linalg.norm(np.array([x,y]) - np.array(p.getBasePositionAndOrientation(u)[0][:2])) < 0.10 for u in self.cube_ids):
                    pos = [x, y, size/2.0 + 0.05]
                    valid = True
                    break
            
            # Fallback (Safety Grid)
            if not valid: 
                # Just place them in a line if random fails
                offset = len(self.cube_ids) * 0.12
                pos = [0.4, 0.2 + offset, size/2.0 + 0.05]

            # Create Body
            uid = p.createMultiBody(
                0.1, 
                p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3), 
                p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=color), 
                basePosition=pos
            )
            
            p.changeDynamics(
                uid, -1, 
                lateralFriction=0.9, 
                ccdSweptSphereRadius=0.002, 
                contactStiffness=10000.0, 
                contactDamping=1.0
            )
            
            self.cube_ids.append(uid)
            self.id_to_size[uid] = size 

        for _ in range(50): p.stepSimulation()
        return self._get_obs()

    def step(self, action):
        p.setJointMotorControlArray(self.robot_id, range(7), p.POSITION_CONTROL, targetPositions=action[:7])
        fp = action[7] * 0.04
        p.setJointMotorControlArray(self.robot_id, [9, 10], p.POSITION_CONTROL, targetPositions=[fp, fp], forces=[200, 200])
        p.stepSimulation()
        if self.dt < 0.01: time.sleep(self.dt)
        return self._get_obs()

    def _get_obs(self):
        flat_cubes = []
        # Because we shuffled 'current_episode_cubes' in reset(), 
        # 'self.cube_ids' is now in that shuffled order.
        # The Network receives: [CubeA_Pos, ..., CubeA_Size, CubeB_Pos, ..., CubeB_Size]
        # where A and B could be any size.
        for uid in self.cube_ids:
            pos, orn = p.getBasePositionAndOrientation(uid)
            size = self.id_to_size[uid]
            flat_cubes.extend(list(pos) + list(orn) + [size])
            
        joint_states = [x[0] for x in p.getJointStates(self.robot_id, range(7))]
        return {"cubes": flat_cubes, "robot": joint_states}