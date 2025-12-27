import numpy as np
import pybullet as p

class StackingExpert:
    def __init__(self, env):
        self.env = env
        # Center of fingers (Offset 0.0)
        self.GRIPPER_LENGTH = 0.0 
        self.stack_target_pos = [0.5, -0.4, 0.0] 
        
    def _solve_ik(self, target_pos, target_orn):
        wrist_target = [
            target_pos[0],
            target_pos[1],
            target_pos[2] + self.GRIPPER_LENGTH
        ]
        joint_poses = p.calculateInverseKinematics(
            self.env.robot_id, 11, wrist_target, target_orn,
            maxNumIterations=100, residualThreshold=1e-5
        )
        return list(joint_poses[:7])

    def _get_grip_value(self, cube_size):
        # Gentle Grip Logic
        # Target = Half Width - 0.5mm
        target_finger_pos = cube_size / 2.0
        target_finger_pos -= 0.0005 
        val = target_finger_pos / 0.04
        return max(0.0, min(1.0, val))

    def generate_episode(self):
        trajectory = []
        current_stack_height = 0.0
        gripper_orn = p.getQuaternionFromEuler([np.pi, 0, 0]) 
        
        # --- FIXED LOGIC HERE ---
        # The env no longer has a fixed 'cube_sizes' list because they are randomized.
        # We must look up the size of each spawned cube ID.
        
        cubes_to_stack = []
        for uid in self.env.cube_ids:
            # Safety check: ensure ID exists in the map
            if uid in self.env.id_to_size:
                size = self.env.id_to_size[uid]
                cubes_to_stack.append((uid, size))
            else:
                print(f"⚠️ Warning: Cube ID {uid} has no size mapping!")

        # SORT BY SIZE (Descending)
        # This ensures the robot always picks Big -> Medium -> Small
        # regardless of the random order they appeared in the list.
        cubes_to_stack.sort(key=lambda x: x[1], reverse=True)

        print(f"   [Planner] Planning stack order for sizes: {[f'{c[1]:.3f}' for c in cubes_to_stack]}")

        for cube_id, cube_size in cubes_to_stack:
            grip_val = self._get_grip_value(cube_size)
            
            cube_pos, _ = p.getBasePositionAndOrientation(cube_id)
            cube_pos = list(cube_pos)
            
            # --- WAYPOINTS ---
            wp_pre_grasp = [cube_pos[0], cube_pos[1], cube_pos[2] + 0.10]
            wp_grasp = [cube_pos[0], cube_pos[1], cube_pos[2]]
            
            # --- PICK SEQUENCE ---
            self._add_path(trajectory, wp_pre_grasp, gripper_orn, gripper=1.0)
            self._add_path(trajectory, wp_grasp, gripper_orn, gripper=1.0)
            # Gentle Close
            self._add_path(trajectory, wp_grasp, gripper_orn, gripper=grip_val, steps=50)
            # Lift
            self._add_path(trajectory, wp_pre_grasp, gripper_orn, gripper=grip_val)
            
            # --- PLACE SEQUENCE ---
            place_z = (cube_size / 2.0) + current_stack_height
            target_pos = [self.stack_target_pos[0], self.stack_target_pos[1], place_z]
            wp_pre_place = [target_pos[0], target_pos[1], place_z + 0.10]
            
            self._add_path(trajectory, wp_pre_place, gripper_orn, gripper=grip_val)
            self._add_path(trajectory, target_pos, gripper_orn, gripper=grip_val)
            # Release
            self._add_path(trajectory, target_pos, gripper_orn, gripper=1.0, steps=30)
            # Retreat
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