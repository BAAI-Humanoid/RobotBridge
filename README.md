# RobotBridge

🤖 Unified Sim2Sim and Sim2Real Deployment Framework for Humanoid Robots - Plug and Play

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📖 Table of Contents

- [Features](#features)
- [Supported Robots](#supported-robots)
- [System Architecture](#system-architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Guide](#usage-guide)
- [Configuration](#configuration)
- [Real Robot Deployment](#real-robot-deployment)
- [Policy Switching](#policy-switching)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## ✨ Features

- 🎯 **Unified Interface**: Same code runs in simulation and on real robots
- 🔄 **Policy Switching**: Real-time smooth switching between locomotion and mimic policies
- 🎮 **Multiple Control Methods**: Keyboard, joystick, and programmatic control
- 🤖 **Multi-Robot Support**: Unitree G1, H1, H1-2, Adam Lite, and more
- 📊 **Visualization Tools**: Built-in MuJoCo visualization and motion markers
- 🔧 **Flexible Configuration**: Hydra-based configuration management system
- 📈 **Smooth Interpolation**: Automatic pose interpolation during policy transitions

## 🤖 Supported Robots

| Robot | DoF | Config File | Status |
|-------|-----|------------|--------|
| Unitree G1 (12 DoF) | 12 | `g1_12dof.yaml` | ✅ Supported |
| Unitree G1 (29 DoF) | 29 | `g1_29dof.yaml` | ✅ Supported |
| Unitree H1 | 19 | `h1_19dof.yaml` | ✅ Supported |
| Unitree H1-2 | 27 | `h1_2_27dof_anneal_21dof.yaml` | ✅ Supported |
| Adam Lite AGX | 23 | `adam_lite_agx_23dof.yaml` | ✅ Supported |

## 🏗️ System Architecture

```
RobotBridge/
├── deploy/                      # Policy Layer
│   ├── agents/                  # Agent implementations
│   │   ├── base_agent.py
│   │   ├── level_agent.py
│   │   ├── mosaic_agent.py
│   │   └── loco_mimic_agent.py  # Policy switching agent
│   ├── envs/                    # Environment implementations
│   │   ├── base_env.py
│   │   ├── level_locomotion.py
│   │   ├── mosaic.py
│   │   └── loco_mimic_switch.py # Policy switching environment
│   ├── simulator/               # Simulator interfaces
│   │   ├── mujoco.py            # MuJoCo simulation
│   │   └── real_world.py        # Real robot interface
│   ├── config/                  # Configuration files
│   └── utils/                   # Utility functions
├── unitree_sdk2/                # Transition Layer
│   ├── trans.cpp                # C++ communication layer
│   └── lcm_types/               # LCM message definitions
└── external/                    # External dependencies
```

## 📦 Installation

### Prerequisites

- Python 3.8+
- GCC/G++ compiler (required for real robot deployment)
- CUDA (optional, for GPU acceleration)

### Step 1: Create Conda Environment

```bash
conda create -n rb python=3.8 -y
conda activate rb
```

### Step 2: Install Dependencies

```bash
# Clone the repository
git clone https://github.com/hitsunzhenguo/RobotBridge.git
cd RobotBridge

# Install Python dependencies
pip install -r requirements.txt

# If you encounter MuJoCo issues on Linux, export this library
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

### Step 3: Compile Transition Layer (Real Robot Only)

```bash
cd unitree_sdk2
find . -exec touch -c {} \;  # Update file timestamps
mkdir -p build && cd build
cmake ..
make
```

After compilation, you'll find the `trans` executable in the `bin/` directory.

## 🚀 Quick Start

### Simulation Testing

#### 1. LEVEL Locomotion

```bash
cd deploy
python run.py --config-name=level_locomotion
```

**Controls:**
- `W/S`: Move forward/backward
- `A/D`: Strafe left/right
- `Q/E`: Rotate left/right
- `Space`: Stop

#### 2. Mosaic (Motion Imitation)

```bash
cd deploy
python run.py --config-name=mosaic \
    env.config.motion.motion_path=data/motion/your_motion.npz
```

#### 3. Locomotion + Mimic Policy Switching

```bash
cd deploy
python run.py --config-name=loco_mimic
```

**Controls:**
- `W/A/S/D/Q/E`: Movement control (in locomotion mode)
- `L`: Switch to Locomotion policy
- `K`: Switch to Mimic policy
- `Space`: Stop movement

## 📚 Usage Guide

### Configuration File Organization

RobotBridge uses Hydra for configuration management with a modular design:

```yaml
# config/loco_mimic.yaml
defaults:
  - _self_
  - robot: g1_29dof  # Full 29-DoF robot configuration
  - obs: level       # Observation configuration (will be overridden per policy)
  - sim: mujoco      # mujoco or real_world
  - agent: loco_mimic
  - env: loco_mimic_switch
  - mimic: mosaic
  - locomotion: level
  - teleop: teleop

device: 'cuda'
```

### Command Line Parameter Override

Use Hydra syntax to override configurations:

```bash
# Change device
python run.py --config-name=loco_mimic device=cuda

# Change robot
python run.py --config-name=loco_mimic \
    asset=h1_19dof \
    control=h1_19dof \
    robot=h1_19dof

# Change motion file
python run.py --config-name=mosaic \
    env.config.motion.motion_path=data/motion/custom_motion.npz

# Disable visualization markers
python run.py --config-name=level_locomotion \
    simulator.config.marker=false
```

### Creating Custom Configurations

#### 1. Create Observation Configuration

Create a configuration file in `config/obs/`:

```yaml
# config/obs/my_obs.yaml
obs_auxiliary:
  - root_quat
  - base_lin_vel
  - base_ang_vel
  - dof_pos
  - dof_vel

obs_dims:
  root_quat: 4
  base_lin_vel: 3
  base_ang_vel: 3
  dof_pos: 29
  dof_vel: 29
```

#### 2. Create Agent Configuration

Create a configuration file in `config/agent/`:

```yaml
# config/agent/my_agent.yaml
_target_: agents.level_agent.LevelAgent
device: ${device}
checkpoint: path/to/your/policy.pt
```

## 🤖 Real Robot Deployment

### Step 1: Launch Transition Layer

On the robot, execute:

```bash
cd unitree_sdk2/build/bin

# Check network interface
ifconfig  # Find the interface name corresponding to 192.168.123.164 (e.g., eth0 or eth1)

# Launch transition layer
./trans_wo_lock eth0

# Press ENTER to trigger communication
```

### Step 2: Launch Policy Layer

In another terminal or remote host:

```bash
cd deploy
python run.py --config-name=loco_mimic \
    simulator=real_world \
    device=cpu
```

**Important Notes:**
- ⚠️ Must launch transition layer before policy layer
- ⚠️ Ensure network connection is working (192.168.123.x subnet)
- ⚠️ Test in a safe environment for first deployment
- ⚠️ Have an emergency stop button ready

### Step 3: Joystick Control

On the real robot, use the Unitree joystick:

**Stick Controls:**
- Left stick: Forward/backward and left/right movement (vx, vy)
- Right stick: Rotation (yaw)

**Policy Switching Buttons:**
- `L1` (left upper button): Switch to Locomotion policy
- `L2` (left lower left button): Switch to Mimic policy

## 🔄 Policy Switching

### Feature Description

RobotBridge supports smooth switching between locomotion and mimic policies at runtime:

- **Locomotion Policy**: Controls robot movement (12 DoF lower body)
- **Mimic Policy**: Executes pre-recorded full-body motions (29 DoF)
- **Smooth Interpolation**: Automatic interpolation during transitions to avoid sudden pose changes
- **Automatic Parameter Switching**: PD control parameters (kp/kd) adjust automatically with policy

### Switching Flow

```
Locomotion → Mimic:
  1. Press K key (sim) or L2 button (real)
  2. System resets motion to first frame
  3. Interpolate 75 steps to mimic initial pose
  4. Switch PD parameters to mimic parameters
  5. Start executing mimic motion

Mimic → Locomotion:
  1. Press L key (sim) or L1 button (real), or automatic when motion finishes
  2. Delay 25 steps before interpolation
  3. Interpolate 75 steps to locomotion pose
  4. Switch PD parameters to locomotion parameters
  5. Resume locomotion control
```

### Configuration Parameters

Adjust interpolation duration in `agents/loco_mimic_agent.py`:

```python
class PolicyInterpManager:
    # [start_delay, interpolation_steps, end_delay]
    DURATIONS_LOCO_MIMIC = [0, 75, 25]   # locomotion -> mimic
    DURATIONS_MIMIC_LOCO = [25, 75, 0]   # mimic -> locomotion
```

### Programmatic Switching

You can also trigger switches programmatically in code:

```python
# During agent runtime
agent.switch_to_mimic()      # Switch to mimic
agent.switch_to_locomotion() # Switch to locomotion
```

## 🐛 Troubleshooting

### Common Issues

#### 1. MuJoCo Display Issues

**Issue**: `GLEW initialization error`

**Solution**:
```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

#### 2. Real Robot Connection Failed

**Issue**: LCM messages not received

**Solutions**:
- Check network connection: `ping 192.168.123.164`
- Confirm transition layer is running and ENTER was pressed
- Use monitor tool: `python deploy/monitor_lcm.py`

#### 3. Joystick Buttons Not Responding

**Issue**: Pressing L1/L2 buttons doesn't switch policies

**Possible Causes and Solutions**:
1. **LCM messages not received**: Run `python deploy/monitor_lcm.py` and press buttons to check for messages
2. **Incorrect button mapping**: Verify correct buttons are used (L1=left_upper_switch, L2=left_lower_left_switch)
3. **Configuration issue**: Confirm using `--config-name=loco_mimic`

#### 4. Abnormal Pose During Policy Switch

**Issue**: Strange pose when switching from locomotion to mimic

**Fixed**: Ensure you're using the latest version. DoF order conversion and default pose issues have been resolved.

#### 5. Import Errors

**Issue**: `ModuleNotFoundError`

**Solution**:
```bash
# Ensure you're in the correct directory
cd deploy
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

#### 6. LCM Issues

**Issue**: robot can't receive tele-operation LCM messages from local PC

**Solutions**:
1. Local PC and robot need to be on the same local area network, you can check by `ping` each other
2. Specify the network port of `239.255.0.0` by running `sudo ip route add 239.255.0.0/16 dev {network port}`
  - You can check network port by running `ifconfig` on Linux
3. If there is already sepcified network port, delete it first by running `sudo ip route del 239.255.0.0/16 dev {old network port}`
4. You can check the result by running `ip route | grep 239`, and if you can see `239.255.0.0/16 dev {network port} scope link`, you succeed.

### Debug Mode

Enable verbose logging:

```python
# Add at the beginning of code
from loguru import logger
import sys

logger.remove()
logger.add(sys.stderr, level="DEBUG")
```

Or set environment variable:
```bash
export LOGURU_LEVEL=DEBUG
python run.py --config-name=loco_mimic
```

### Evaluation Guide
#### Overview
This guide outlines the standard procedure for running model evaluation with automatic checkpoint loading based on file extensions (ONNX/TorchScript).

#### Prerequisites
- Ensure all dependencies are installed as per the project requirements.
- Valid checkpoint files (`.onnx`, `.pt`, `.pth`, `.jit`) and motion data are prepared at `/path/to/data`.

#### Standard Evaluation
1. Modify the `eval.sh` script with the following core configurations (adjust parameters as needed):
   ```bash
   HYDRA_FULL_ERROR=1 python run.py --config-name=eval \
       mimic.policy.checkpoint=/path/to/checkpoint.onnx \
       mimic.policy.use_estimator=False \
       robot.control.viewer=False \
       robot.control.real_time=False \
       mimic.motion.motion_path=/path/to/motion/path \
       mimic.policy.history_length=5 \
       mimic.policy.eval_mode=True \
       mimic.motion.command_horizon=1
   ```
2. Optimize evaluation speed by modifying `deploy/envs/mosaic.py`:
   - Locate the function with `save_video` enabled.
   - Keep only `self.frames = []` and comment out all other related lines.

#### Specialized Evaluation (GMT/Twist)
Use the following commands directly for GMT and Twist model evaluation (adjust motion path if necessary):
```bash
# GMT Evaluation
HYDRA_FULL_ERROR=1 python run.py --config-name=gmt \
    mimic.policy.checkpoint=/path/to/gmt.pt \
    robot.control.viewer=False \
    robot.control.real_time=False \
    mimic.motion.motion_path=/path/to/motion/path

# Twist Evaluation
HYDRA_FULL_ERROR=1 python run.py --config-name=twist \
    mimic.policy.checkpoint=/path/to/twist.pt \
    robot.control.viewer=False \
    robot.control.real_time=False \
    mimic.motion.motion_path=/path/to/motion/path
```

#### Notes
- All file paths (checkpoint/motion data) should be replaced with actual paths in production use.
- `viewer` and `real_time` are set to `False` by default to maximize evaluation efficiency.
- Adjust `history_length` and `command_horizon` according to the actual model configuration.

## 📝 Examples

### Example 1: G1 Robot LEVEL Locomotion

```bash
cd deploy
python run.py --config-name=level_locomotion \
    asset=g1_29dof \
    control=g1_29dof \
    robot=g1_29dof \
    device=cpu
```

### Example 2: H1 Robot Mosaic

```bash
cd deploy
python run.py --config-name=mosaic \
    asset=h1_19dof \
    control=h1_19dof \
    robot=h1_19dof \
    env.config.motion.motion_path=data/motion/h1_motion.npz
```

### Example 3: Policy Switching (Simulation)

```bash
cd deploy
python run.py --config-name=loco_mimic \
    asset=g1_29dof \
    control=g1_29dof \
    robot=g1_29dof \
    device=cpu
```

### Example 4: Real Robot Deployment

```bash
# First, launch transition layer on robot
cd unitree_sdk2/build/bin
./trans_wo_lock eth0

# Then, on policy layer host
cd deploy
python run.py --config-name=loco_mimic \
    simulator=real_world \
    asset=g1_29dof \
    control=g1_29dof \
    robot=g1_29dof \
    device=cpu
```

## 🤝 Contributing

Issues and Pull Requests are welcome!

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- Unitree Robotics for robot hardware and SDK
- MuJoCo for physics simulation
- Hydra for configuration management

## 📞 Contact

For questions or suggestions, please submit an Issue or contact the maintainers.

---

**Note**: Before deploying on real robots, ensure:
1. ✅ Thoroughly tested in simulation environment
2. ✅ Emergency stop measures prepared
3. ✅ First test conducted in safe environment
4. ✅ Familiar with robot operation procedures
