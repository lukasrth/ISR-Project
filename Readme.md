# ISR Project: Cube Stacking with Diffusion Policy

This repository contains the simulation environment and data generation pipeline for the Intelligent Systems and Robotics (ISR) course project. [cite_start]The goal is to train a robot manipulator (Franka Emika Panda) to stack cubes in descending order of size using **Diffusion Policy**[cite: 4].

[cite_start]The project uses **PyBullet** for simulation [cite: 17][cite_start], **PyRoki** for inverse kinematics[cite: 9], and outputs data compatible with the Diffusion Policy training pipeline.

## 📂 Project Structure

```text
ISR-Project/
├── data/                    # Generated .zarr datasets (created automatically)
├── env/                     # PyBullet Environment wrapper
│   └── sim_env.py           # Simulation logic, procedural generation, & sensor wrappers
├── expert/                  # Motion Planner
│   └── motion_planner.py    # Inverse Kinematics & Trajectory Generation
├── generate_data.py         # Main CLI script to collect demonstrations
├── save_replay_video.py     # Utility to verify data validity (renders mp4)
├── requirements.txt         # Dependencies
└── README.md
```
## Setup

We use Conda to manage the Python environment. This setup ensures compatibility between the JAX-based kinematics (PyRoki) and PyTorch-based training.

### 1. Create the conda Environment

We recommend Python 3.9, which is the standard stable version for Diffusion Policy and PyBullet.


```bash
# Initialize conda for the current shell (useful for CI / non-interactive shells)
eval "$(conda shell.bash hook)"

# Create the conda environment
conda create -n ISR_Project python=3.10 -y

# Activate the environment
conda activate ISR_Project
```

### 2. Install Dependencies 


```bash
# Update pip first
pip install --upgrade pip

# Install dependencies from requirements.txt
pip install -r requirements.txt
```
Note for PyRoki: The requirements.txt installs PyRoki directly from GitHub. Ensure you have git installed on your system. If the install fails, verify your JAX installation matches your OS (Linux/macOS/Windows).

### 3 Verify Installation

```bash
# Run a quick Python command to verify installation
python -c 'import pybullet; import pyroki; import torch; print("Setup Complete!")'
```

## Generating Expert Data

To train the Diffusion Policy, we generate a dataset of expert demonstrations. The pipeline uses a scripted expert to move the robot while recording State (joint angles, object positions) and optionally Vision (camera images).

### 1.  Run Data Collection

Execute the generation script. This will launch a PyBullet GUI (optional) and save the trajectories.

```bash
python generate_data.py
```

- Flag with --num_episodes to modify the number of expert runs (default 50)
- Flag with --visual to save images for realistic training  
- GUI Mode: By default, the script runs with gui=True so you can watch the robot.
- Headless Mode: To generate data faster, edit generate_data.py and set gui=False in the StackingEnv initialization.

### 2. Output

The script produces a Zarr file located at data/stacking_demo.zarr.

File Structure:

```text
data/stacking_demo.zarr/
├── data/
│   ├── action          (Shape: [N, 8])   # 7 Joint Positions + 1 Gripper State
│   ├── state           (Shape: [N, 31])  # 7 Robot Joints + 21 Cube Pos/Orn
│   └── img             (Shape: [N, 96, 96, 3])  # (Optional) RGB Frames
└── meta/
    └── episode_ends    (Shape: [Num_Episodes])  # Indices marking end of trajectories
```

You can check the output with inspect_data.py and replay_data.py to data assessing and visual inspection. 

## Next Steps: Training

Once data/stacking_demo.zarr is generated:

Next Steps: Training

1.  Clone the Diffusion Policy repository.

2.   Create a configuration file (YAML) pointing to your data/stacking_demo.zarr.

3.   For State Policy: Use data/state (31-dim) as input.

4.   For Visual Policy: Use data/img (96x96x3) + data/state (7-dim robot joints) as input.

## Troubleshooting

- JAX/CUDA errors: If you have GPU conflicts between JAX and PyTorch, you can force JAX to use CPU only (sufficient for kinematics) by running:
```Bash
export JAX_PLATFORM_NAME=cpu
```
before running the python script.
