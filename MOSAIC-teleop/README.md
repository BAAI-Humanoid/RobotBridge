# MOSAIC-teleop

**MOSAIC-teleop** is a versatile teleoperation framework adapted for the **MOSAIC** ecosystem. It enables high-fidelity control of humanoid robots (Unitree G1 now) by supporting dual input modalities: **Inertial Motion Capture** (Noitom) and **Virtual Reality** (PICO 4).

This framework bridges human motion to robot actions with low latency, suitable for data collection, imitation learning, and real-time remote control tasks.

## 🌟 Features

- **Dual Modality Support**:
  - 🦾 **Inertial Mocap**: Seamless integration with Noitom Hi5/Hi7 gloves and body suits for precise finger and body tracking.
  - 🥽 **VR Control**: Native support for PICO 4 headsets, allowing immersive teleoperation via controllers or hand tracking.
- **General Motion Retargeting (GMR)**: Integrated robust retargeting algorithms to map human kinematics to robot joint spaces efficiently.
- **Real-time Streaming**: Optimized LCM-based communication for low-latency command transmission.
- **Data Recording**: Built-in tools to record synchronized motion trajectories for offline training and analysis.
- **Extensible Architecture**: Modular design allows easy adaptation to other robots or input devices.

## 🚀 Quick Start

### 1. Environment Setup

We recommend using **Conda** to manage dependencies. Ensure you have Python 3.10 installed.

```bash
conda create -n teleop python=3.10 -y
conda activate teleop
```

Please install the dependencies by referring to https://github.com/YanjieZe/GMR. Note: Ensure all dependencies are installed within the teleop conda environment."

```bash
# Install project dependencies
# Make sure you are in the root directory of MOSAIC-teleop
pip install -r requirements.txt
```

### 3. Running the Teleoperation

Choose the script corresponding to your input device:

#### Option A: Noitom Inertial Motion Capture
If you are using a Noitom mocap suit/gloves:

```bash
bash teleop_noitom.sh
```

#### Option B: PICO 4 VR
If you are using a PICO 4 headset:

```bash
bash teleop_vr.sh
```

> **Tip**: You can inspect the `.sh` files to see specific command-line arguments being passed (e.g., `--robot`, `--record-dir`). You can modify these scripts or pass additional arguments directly to the python scripts if needed.

## 📂 Project Structure

```text
MOSAIC-teleop/
├── assets/                 # Robot URDFs, meshes, and configuration files
├── deploy/                 # Deployment scripts and main entry points
│   ├── axisstudio_to_gmr_retarget.py  # Core logic for Noitom retargeting
│   └── ...
├── general_motion_retargeting/ # GMR algorithm source code
├── online/                 # Real-time adapters and API wrappers
│   ├── mocap_robotapi.py   # AxisStudio/Mocap API interface
│   └── realtime_adapter.py # VR/Realtime data adapters
├── teleop_noitom.sh        # Launcher script for Noitom mode
├── teleop_vr.sh            # Launcher script for VR mode (Update this filename if different)
└── README.md
```

## ⚙️ Advanced Usage

### Recording Data
To record motion data for imitation learning, add the `--record` flag (if supported by the script) or configure the `--record-dir` argument:

```bash
python deploy/axisstudio_to_gmr_retarget.py \
  --robot unitree_g1 \
  --record-dir ./my_recordings \
  --target-fps 50 \
  --record-prefix demo_trial
```

### Customizing Robot Configuration
Modify the `assets/` folder to swap URDF files if you are adapting this for a different robot morphology (e.g., Unitree H1, Tesla Optimus).

## 🛠 Troubleshooting

- **ImportError: No module named 'online'**:
  Ensure you are running the script from the project root directory. If the issue persists, try adding the root path to your `PYTHONPATH`:
  ```bash
  export PYTHONPATH=$(pwd):$PYTHONPATH
  ```

- **LCM Connection Failed**:
  Verify that the LCM daemon is running (`lcm-server`) and that your firewall allows UDP traffic on the LCM port.

- **Mocap Data Not Moving**:
  Check if Axis Studio is actively streaming and that the IP address in `online/mocap_robotapi.py` matches your mocap system's broadcast IP.

## 📄 License

This project is released under the No License.

## 🙏 Acknowledgements

- This project, **MOSAIC-teleop**, is adapted from and built upon the **[GMR (General Motion Retargeting)](https://github.com/YanjieZe/GMR)** framework. We sincerely thank the authors for open-sourcing their powerful retargeting algorithms and real-time teleoperation tools.
- Integrates with **Unitree Robotics** SDK.
- Supports **Noitom** and **PICO** hardware ecosystems.






