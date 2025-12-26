# ISR Project: Cube Stacking with Diffusion Policy

This repository contains the simulation environment and data generation pipeline for the Intelligent Systems and Robotics (ISR) course project. [cite_start]The goal is to train a robot manipulator (Franka Emika Panda) to stack cubes in descending order of size using **Diffusion Policy**[cite: 4].

[cite_start]The project uses **PyBullet** for simulation [cite: 17][cite_start], **PyRoki** for inverse kinematics[cite: 9], and outputs data compatible with the Diffusion Policy training pipeline.

## 📂 Project Structure

```text
ISR-Project/
├── assets/                  # Robot and Object URDFs
│   ├── cube_large.urdf
│   ├── cube_medium.urdf
│   ├── cube_small.urdf
│   └── table.urdf
├── data/                    # Generated .zarr datasets (ignored by git)
├── env/                     # PyBullet Environment wrapper
│   └── sim_env.py
├── expert/                  # Motion Planner (PyRoki/IK)
│   └── motion_planner.py
├── generate_data.py         # Main script to collect demonstrations
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

To train the Diffusion Policy, we first need a dataset of expert demonstrations. This pipeline uses a scripted expert (Finite State Machine) to generate "perfect" stacking trajectories.

### 1. Configure the Environment

Place the URDF files in the assets/ folder. The data collection script expects these filenames and sizes:

- assets/cube_large.urdf — 5 cm edge (0.05 m)
- assets/cube_medium.urdf — 4 cm edge (0.04 m)
- assets/cube_small.urdf — 3 cm edge (0.03 m)





### 2. Run Data Collection

Execute the generation script. This will launch a PyBullet GUI (optional) and save the trajectories.

```bash
python generate_data.py
```

- GUI Mode: By default, the script runs with gui=True so you can watch the robot.

- Headless Mode: To generate data faster, edit generate_data.py and set gui=False in the StackingEnv initialization.

### 3. Output

The script produces a Zarr file located at: data/stacking_demo.zarr

This file contains:

-  data/action: Joint positions/velocities and gripper state.
-  data/state: Robot joint angles and cube positions.
-  meta/episode_ends: Indices separating valid episodes.

You can check the output with inspect_data.py and replay_data.py to data assessing and visual inspection. 

## Next Steps: Training

Once data/stacking_demo.zarr is generated:

- Clone the Diffusion Policy repository.
-  Create a configuration file (YAML) that points to your data/stacking_demo.zarr path.
- Run the training script. You do not need to run PyBullet during training.
- The policy will learn to map the state observations directly to actions.

## Troubleshooting

- Missing URDFs: If PyBullet crashes, check that the assets/ folder contains valid URDF files for the cubes.
- JAX/CUDA errors: If you have GPU conflicts between JAX and PyTorch, you can force JAX to use CPU only (sufficient for kinematics) by running:
```Bash
export JAX_PLATFORM_NAME=cpu
```
before running the python script.
