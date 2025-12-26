import numpy as np
import pybullet as p

class StackingExpert:
    def __init__(self, env):
        self.env = env
        # Link 11 is the center point between fingers.
        # Offset 0.0 puts the center of the gripper at the center of the cube.
        self.GRIPPER_LENGTH = 0.0
        
        # Target: Slightly to the right of the robot
        self.stack_target_pos = [0.5, -0.4, 0.0]
        
    def _solve_ik(self, target_pos, target_orn):
        wrist_target = [
            target_pos[0],
            target_pos[1],
            target_pos[2] + self.GRIPPER_LENGTH
        ]
        # Use Link 11 (TCP)
        joint_poses = p.calculateInverseKinematics(
            self.env.robot_id, 11, wrist_target, target_orn,
            maxNumIterations=100, residualThreshold=1e-5
        )
        return list(joint_poses[:7])

    def _get_grip_value(self, cube_size):
        """
        Calculates gentle grip value.
        Panda Gripper: 
           0.04 = Open (8cm width)
           0.00 = Closed (0cm width)
        
        Formula: Joint_Pos = (Cube_Width / 2) - Squeeze_Margin
        """
        # 1. Determine Half-Width of the object
        target_finger_pos = cube_size / 2.0
        
        # 2. Apply TINY squeeze (0.5mm)
        # Just enough to trigger the friction contact, not enough to clip.
        target_finger_pos -= 0.0005 
        
        # 3. Normalize to 0..1 range (where 1.0 corresponds to 0.04m joint limit)
        # Note: Your sim_env.py multiplies action * 0.04.
        # So we return the value such that: action * 0.04 = target_finger_pos
        val = target_finger_pos / 0.04
        
        return max(0.0, min(1.0, val))

    def generate_episode(self):
        trajectory = []
        current_stack_height = 0.0
        gripper_orn = p.getQuaternionFromEuler([np.pi, 0, 0])
        
        print(f"   [Planner] Found {len(self.env.cube_ids)} cubes.")

        for i, cube_id in enumerate(self.env.cube_ids):
            cube_size = self.env.cube_sizes[i]
            grip_val = self._get_grip_value(cube_size)
            
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            cube_pos = list(cube_pos)
            
            # --- WAYPOINTS ---
            # Hover 10cm above
            wp_pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.10]
            # Grasp Center
            wp_grasp = [cube_pos[0], cube_pos[1], cube_pos[2]]
            
            # --- SEQUENCE ---
            
            # 1. Approach (Open)
            self._add_path(trajectory, wp_pre_grasp, gripper_orn, gripper=1.0)
            self._add_path(trajectory, wp_grasp, gripper_orn, gripper=1.0)
            
            # 2. Grasp (Gentle) - Wait longer (50 steps) for friction to bite
            self._add_path(trajectory, wp_grasp, gripper_orn, gripper=grip_val, steps=50)
            
            # 3. Lift (Gentle Hold)
            self._add_path(trajectory, wp_pre_grasp, gripper_orn, gripper=grip_val)
            
            # --- PLACE ---
            place_z = (cube_size / 2.0) + current_stack_height
            target_pos = [self.stack_target_pos[0], self.stack_target_pos[1], place_z]
            wp_pre_place = [target_pos[0], target_pos[1], place_z + 0.10]
            
            # 4. Transport
            self._add_path(trajectory, wp_pre_place, gripper_orn, gripper=grip_val)
            self._add_path(trajectory, target_pos, gripper_orn, gripper=grip_val)
            
            # 5. Release
            self._add_path(trajectory, target_pos, gripper_orn, gripper=1.0, steps=30)
            
            # 6. Retreat
            self._add_path(trajectory, wp_pre_place, gripper_orn, gripper=1.0)
            
            current_stack_height += cube_size

        return np.array(trajectory)

    def _add_path(self, trajectory_list, target_pos, target_orn, gripper, steps=40):
        if not trajectory_list:
            current_joints = [s[0] for s in p.getJointStates(self.env.robot_id, range(7))]
        else:
            current_joints = trajectory_list[-1][:7]
            
        target_joints = self._solve_ik(target_pos, target_orn)
        
        for step in range(1, steps + 1):
            alpha = step / steps
            interp_joints = (1 - alpha) * np.array(current_joints) + alpha * np.array(target_joints)
            action = np.concatenate([interp_joints, [gripper]])
            trajectory_list.append(action)