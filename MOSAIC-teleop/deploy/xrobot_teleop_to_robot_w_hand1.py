"""
conda activate gmr
sudo ufw disable
python xrobot_teleop_to_robot_w_hand.py --robot unitree_g1

State Machine Controls:
- Right controller key_one (A): Start/stop motion recording
- Left controller key_one (X): Exit program from any state
- Left controller axis_click: Emergency stop - kills sim2real.sh process
- Left controller trigger: LEFT dex3 grip amount (0=open, 1=closed)
- Right controller trigger: RIGHT dex3 grip amount (0=open, 1=closed)
- Left controller axis: Control root xy velocity and yaw velocity
- Right controller axis: Fine-tune root xy velocity and yaw velocity
- Auto-transition: idle -> teleop when motion data is available

States:
- idle: Waiting for input or data
- teleop: Processing motion retargeting with velocity control
- exit: Program will terminate

Whole-Body Teleop Features:
- Publishes retargeted body pose to LCM (camera_reference_data)
- Uses retargeted motion directly from the teleoperation stream
"""
import argparse
import subprocess
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mujoco as mj
import mujoco.viewer as mjv
import numpy as np
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation as R
from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting.robot_motion_viewer import draw_frame
from general_motion_retargeting import ROBOT_XML_DICT, ROBOT_BASE_DICT
from rich import print
from tqdm import tqdm
import cv2
from rich import print
from general_motion_retargeting.xrobot_utils import XRobotStreamer
from lcm_publisher import LCMPublisher, DexCommandPublisher

from data_utils.fps_monitor import FPSMonitor

_G1_LCM_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

_LEFT_FOOT_CANDIDATES = ("Left_Foot", "LeftFoot", "LeftFootMod", "left_foot", "leftFoot")
_RIGHT_FOOT_CANDIDATES = ("Right_Foot", "RightFoot", "RightFootMod", "right_foot", "rightFoot")

class StateMachine:
    def __init__(self):
        """
        State process for teleoperation:
        idle -> teleop -> exit
        """
        self.state = "idle"
        self.previous_state = "idle"
        self.right_record_key_was_pressed = False
        self.left_exit_key_was_pressed = False
        self.left_axis_click_was_pressed = False
        self.record_toggle_requested = False
        # Velocity commands from joystick
        self.velocity_commands = np.array([0.0, 0.0, 0.0])  # [vx, vy, vyaw]

    def update(self, controller_data):
        """Update state machine with controller data"""
        # Store previous state
        self.previous_state = self.state
        
        # Get current button states
        right_key_current = controller_data.get('RightController', {}).get('key_one', False)
        left_key_current = controller_data.get('LeftController', {}).get('key_one', False)
        
        # Emergency stop - left controller axis_click
        left_axis_click_current = controller_data.get('LeftController', {}).get('axis_click', False)

        # Detect button presses
        right_key_just_pressed = right_key_current and not self.right_record_key_was_pressed
        left_key_just_pressed = left_key_current and not self.left_exit_key_was_pressed
        left_axis_click_just_pressed = left_axis_click_current and not self.left_axis_click_was_pressed

        # Handle left axis click - emergency stop
        if left_axis_click_just_pressed:
            self._emergency_stop()

        # Handle left key press - exit from any state
        if left_key_just_pressed:
            self.state = "exit"

        # Handle right key press - request recording toggle
        elif right_key_just_pressed:
            self.record_toggle_requested = True

        # Extract velocity commands from controller axes
        self._update_velocity_commands(controller_data)
        
        # Update button state tracking
        self.right_record_key_was_pressed = right_key_current
        self.left_exit_key_was_pressed = left_key_current
        self.left_axis_click_was_pressed = left_axis_click_current
    
    def _update_velocity_commands(self, controller_data):
        """Update velocity commands from controller axes"""
        left_axis = controller_data.get('LeftController', {}).get('axis', [0.0, 0.0])
        right_axis = controller_data.get('RightController', {}).get('axis', [0.0, 0.0])
        
        # Use left stick for xy movement, right stick for yaw rotation
        if len(left_axis) >= 2 and len(right_axis) >= 2:
            # Scale factors for velocity commands
            xy_scale = 2.0  # m/s
            yaw_scale = 3.0  # rad/s
            
            self.velocity_commands[0] = left_axis[1] * xy_scale   # forward/backward (y axis inverted)
            self.velocity_commands[1] = -left_axis[0] * xy_scale  # left/right (x axis inverted)
            self.velocity_commands[2] = -right_axis[0] * yaw_scale  # yaw rotation (x axis inverted)
    
    def has_state_changed(self):
        """Check if state has changed since last update"""
        return self.state != self.previous_state

    def consume_record_toggle(self):
        """Consume a pending record toggle request."""
        if self.record_toggle_requested:
            self.record_toggle_requested = False
            return True
        return False
    
    def get_current_state(self):
        return self.state
    

    def get_velocity_commands(self):
        return self.velocity_commands.copy()
        
    def is_teleop_active(self):
        """Return True if currently in teleop state"""
        return self.state == "teleop"

    def auto_transition(self, has_data: bool):
        """Auto transition from idle -> teleop when data is available."""
        if self.state == "idle" and has_data:
            self.state = "teleop"
        
    def should_exit(self):
        """Return True if should exit the program"""
        return self.state == "exit"
    
    def _emergency_stop(self):
        """Emergency stop: kill sim2real.sh process (server_low_level_g1_real_future.py)"""
        try:
            print("[EMERGENCY STOP] Killing sim2real.sh process...")
            # Kill sim2real.sh which contains server_low_level_g1_real_future.py
            result = subprocess.run(['pkill', '-f', 'sim2real.sh'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed sim2real.sh process")
            else:
                print(f"[EMERGENCY STOP] pkill returned code {result.returncode}")

            # Also try to kill the specific server script directly as backup
            result2 = subprocess.run(['pkill', '-f', 'server_low_level_g1_real_future.py'], 
                                   capture_output=True, text=True, timeout=5)
            if result2.returncode == 0:
                print("[EMERGENCY STOP] Successfully killed server_low_level_g1_real_future.py process")
            else:
                print(f"[EMERGENCY STOP] pkill for server script returned code {result2.returncode}")
                
        except subprocess.TimeoutExpired:
            print("[EMERGENCY STOP] pkill command timed out")
        except Exception as e:
            print(f"[EMERGENCY STOP] Error executing pkill: {e}")

class XRobotTeleopToRobot:
    def __init__(self, args):
        self.args = args
        self.robot_name = args.robot
        self.xml_file = ROBOT_XML_DICT[args.robot]
        self.robot_base = ROBOT_BASE_DICT[args.robot]
        self.headless = args.headless
        
        # Initialize state tracking
        self.last_qpos = None
        self.target_fps = args.target_fps

        # Initialize components
        self.teleop_data_streamer = None
        self.retarget = None
        self.model = None
        self.data = None
        self.lcm_pub = None
        self.lcm_qpos_indices = None
        self.default_qpos = None
        self.state_machine = StateMachine()
        self.rate = None
        
        # Video recording
        self.video_writer = None
        self.renderer = None

        # Motion recording
        self.record_dir = Path(args.record_dir) if args.record_dir else None
        self.record_enabled = self.record_dir is not None
        self.record_running = bool(self.record_enabled and args.record_auto_start)
        self.record_toggle = False
        self.record_failed = False
        self.record_saved = False
        self.record_buffer = []
        self.record_thread = None
        self.stop_listen = None

        # Gripper control (dex3) using analog triggers
        self.left_grip_state = 0.0  # 0.0=open, 1.0=closed
        self.right_grip_state = 0.0
        self.last_grip_publish_us = 0
        self.dex_pub = None
        
        # FPS monitoring
        self.fps_monitor = FPSMonitor(
            enable_detailed_stats=args.measure_fps,
            quick_print_interval=100,
            detailed_print_interval=1000,
            expected_fps=self.target_fps,
            name="Teleop Loop"
        )

        # Ground offset estimator (contact-gated)
        self._ground_initialized = False
        self._ground_init_samples = []
        self._ground_z_est = 0.0
        self._prev_left_z = None
        self._prev_right_z = None
        self._prev_ground_time = None


    def setup_teleop_data_streamer(self):
        """Initialize and start the teleop data streamer"""
        self.teleop_data_streamer = XRobotStreamer()
        print("Teleop data streamer initialized")
        
    def setup_lcm_publisher(self):
        """Setup LCM publisher for real robot deployment"""
        self.lcm_pub = LCMPublisher(channel_name=self.args.lcm_channel)
        print(f"LCM publisher initialized: {self.args.lcm_channel}")

    def setup_gripper_publisher(self):
        """Setup LCM publisher for dex3 gripper commands."""
        self.dex_pub = DexCommandPublisher()
        if self.dex_pub.lc:
            print("Gripper LCM publisher initialized: dex_command")
        else:
            print("[gripper] dex_command_lcmt unavailable; skip gripper LCM publisher.")

    def _build_lcm_qpos_indices(self):
        """Build joint index mapping for LCM publishing"""
        if self.retarget is None:
            return
        joint_name_to_qpos = {}
        for jid in range(int(self.retarget.model.njnt)):
            jname = self.retarget.model.joint(jid).name
            adr = int(self.retarget.model.jnt_qposadr[jid])
            joint_name_to_qpos[jname] = adr

        qpos_indices = []
        missing = []
        for name in _G1_LCM_JOINT_ORDER:
            if name in joint_name_to_qpos:
                qpos_indices.append(joint_name_to_qpos[name])
                continue
            base = name[:-len("_joint")] if name.endswith("_joint") else name
            if base in joint_name_to_qpos:
                qpos_indices.append(joint_name_to_qpos[base])
                continue
            missing.append(name)
            qpos_indices.append(None)
        if missing:
            print(f"[LCM] Warning: {len(missing)} joints not found in model: {missing[:5]}")
        self.lcm_qpos_indices = qpos_indices

    def setup_retargeting_system(self):
        """Initialize the motion retargeting system"""
        self.retarget = GMR(
            src_human="xrobot",
            tgt_robot="unitree_g1",
            actual_human_height=self.args.actual_human_height,
        )
        print("Retargeting system initialized")
    
    def setup_mujoco_simulation(self):
        """Setup MuJoCo model and data"""
        self.model = mj.MjModel.from_xml_path(str(self.xml_file))
        self.data = mj.MjData(self.model)
        self.default_qpos = self.data.qpos.copy()
        print("MuJoCo simulation initialized")
        
    def setup_video_recording(self):
        """Setup video recording if requested"""
        if not self.args.record_video:
            return

        if self.headless:
            print("Headless mode: skip video recording setup to reduce overhead")
            return
            
        self.video_writer = cv2.VideoWriter(
            'output.mp4', 
            cv2.VideoWriter_fourcc(*'mp4v'), 
            30, 
            (640, 480)
        )
        width, height = 640, 480
        self.renderer = mj.Renderer(self.model, height=height, width=width)
        print("Video recording setup completed")
        
    def setup_rate_limiter(self):
        """Setup rate limiter for consistent FPS"""
        self.rate = RateLimiter(frequency=self.target_fps, warn=False)
        print(f"Rate limiter setup for {self.target_fps} FPS")

    def setup_recording(self):
        """Setup motion recording if requested."""
        if not self.record_enabled:
            return
        self.record_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Recording enabled: dir={self.record_dir} prefix={self.args.record_prefix}")
        try:
            from sshkeyboard import listen_keyboard, stop_listening

            self.stop_listen = stop_listening

            def on_press(key):
                if key == "s":
                    self.record_toggle = True
                elif key == "f":
                    self.record_failed = True
                    self.record_toggle = True

            self.record_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={"on_press": on_press, "until": None, "sequential": False},
                daemon=True,
            )
            self.record_thread.start()
            if not self.record_running:
                print("  - press 's' to start/stop recording, 'f' to discard")
        except Exception as exc:
            print(f"Warning: sshkeyboard unavailable ({exc}); recording uses auto-start only.")
            self.record_running = bool(self.record_enabled)

    def _build_lcm_payload(self, qpos):
        if self.lcm_pub is None or qpos is None or self.lcm_qpos_indices is None:
            return None
        qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if qpos.shape[0] < 7:
            return None
        root_pos = qpos[:3].tolist()
        root_rot = qpos[3:7].tolist()
        dof_pos = []
        for idx in self.lcm_qpos_indices:
            if idx is None or idx >= qpos.shape[0]:
                dof_pos.append(0.0)
            else:
                dof_pos.append(float(qpos[int(idx)]))
        return dof_pos, root_pos, root_rot

    def _save_record(self):
        if not self.record_enabled or self.record_saved:
            return
        if not self.record_buffer:
            print("[record] No frames captured; skip saving.")
            self.record_saved = True
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.record_dir / f"{self.args.record_prefix}_{ts}.npz"
        J = len(_G1_LCM_JOINT_ORDER)
        dof_pos = np.asarray([row[:J] for row in self.record_buffer], dtype=np.float32)
        root_pos = np.asarray([row[J:J+3] for row in self.record_buffer], dtype=np.float32)
        root_rot = np.asarray([row[J+3:J+7] for row in self.record_buffer], dtype=np.float32)
        timestamp_us = np.asarray([row[J+7] for row in self.record_buffer], dtype=np.int64)
        fps = float(self.target_fps) if self.target_fps else 0.0
        np.savez_compressed(
            out_path,
            dof_pos=dof_pos,
            root_pos=root_pos,
            root_rot=root_rot,
            timestamp_us=timestamp_us,
            joint_names=np.asarray(_G1_LCM_JOINT_ORDER, dtype=object),
            fps=np.float32(fps),
        )
        print(f"[record] Saved {len(self.record_buffer)} frames -> {out_path}")
        self.record_saved = True
        
    def _publish_gripper_command(self, timestamp_us: int):
        if self.dex_pub is None or self.dex_pub.lc is None:
            return
        self.dex_pub.publish(self.left_grip_state, self.right_grip_state, timestamp_us)
        self.last_grip_publish_us = timestamp_us

    def _update_gripper_state(self, controller_data):
        """Use analog triggers to control gripper closure (0.0=open, 1.0=closed)."""
        if controller_data is None:
            return
        def get_trigger(side: str):
            c = controller_data.get(side, {})
            trig = float(c.get('index_trig', 0.0))
            grip = float(c.get('grip', 0.0))
            return max(trig, grip)

        left_val = max(0.0, min(1.0, get_trigger('LeftController')))
        right_val = max(0.0, min(1.0, get_trigger('RightController')))

        changed = (abs(left_val - self.left_grip_state) > 0.02) or (abs(right_val - self.right_grip_state) > 0.02)
        self.left_grip_state = left_val
        self.right_grip_state = right_val

        now_us = int(time.time() * 1e6)
        if changed or (now_us - self.last_grip_publish_us) > 200000:  # at least 5Hz keep-alive
            self._publish_gripper_command(now_us)
        
    def get_teleop_data(self):
        """Get current teleop data from streamer"""
        if self.teleop_data_streamer is not None:
            return self.teleop_data_streamer.get_current_frame()
        return None, None, None, None, None
        
    def process_retargeting(self, smplx_data):
        """Process motion retargeting and return qpos."""
        if smplx_data is None or self.retarget is None:
            return None

        self._update_ground_offset(smplx_data)
        
        # Retarget till convergence
        qpos = self.retarget.retarget(
            smplx_data,
            offset_to_ground=self.args.offset_to_ground,
        )

        # left_pos, right_pos = self._get_scaled_foot_positions(smplx_data)
        # if left_pos is not None and right_pos is not None:
        #     # 获取当前帧人体脚部的原始 Z 值 (scalar)
        #     left_z = float(left_pos[2])
        #     right_z = float(right_pos[2])
            
        #     # GMR 在内部计算 target 时会做如下操作: target_z = human_z - ground_offset
        #     # 我们需要预测 IK 最终尝试触达的最低高度
        #     min_human_z = min(left_z, right_z)
        #     predicted_target_z = min_human_z - self.retarget.ground_offset
            
        #     # 在 Mujoco 仿真中，地面高度通常是 0
        #     robot_ground_z = 0.0 
            
        #     # 计算穿透量 (标量比较)
        #     penetration = robot_ground_z - predicted_target_z
            
        #     # 如果预测值会入地 (penetration > 0)，则将 Root Z 抬升
        #     if penetration > 0:
        #         # qpos[2] 是 Root 的 Z 坐标
        #         qpos[2] += penetration

        self.last_qpos = qpos.copy()
        return qpos

    def _find_body_pos(self, human_data, candidates):
        for name in candidates:
            if name in human_data:
                return human_data[name][0]
        return None

    def _get_scaled_foot_positions(self, smplx_data):
        if self.retarget is None:
            return None, None
        human_data = self.retarget.to_numpy(smplx_data)
        human_data = self.retarget.scale_human_data(
            human_data,
            self.retarget.human_root_name,
            self.retarget.human_scale_table,
        )
        human_data = self.retarget.offset_human_data(
            human_data,
            self.retarget.pos_offsets1,
            self.retarget.rot_offsets1,
        )
        left_pos = self._find_body_pos(human_data, _LEFT_FOOT_CANDIDATES)
        right_pos = self._find_body_pos(human_data, _RIGHT_FOOT_CANDIDATES)
        return left_pos, right_pos

    def _update_ground_offset(self, smplx_data):
        if not self.args.auto_ground:
            return
        if smplx_data is None or self.retarget is None:
            return

        left_pos, right_pos = self._get_scaled_foot_positions(smplx_data)
        if left_pos is None or right_pos is None:
            return

        left_z = float(left_pos[2])
        right_z = float(right_pos[2])
        min_z = min(left_z, right_z)

        now = time.time()
        if self._prev_ground_time is None:
            self._prev_ground_time = now
            self._prev_left_z = left_z
            self._prev_right_z = right_z
            self._ground_init_samples.append(min_z)
            if len(self._ground_init_samples) >= self.args.ground_init_frames:
                self._ground_z_est = float(np.median(self._ground_init_samples))
                self._ground_initialized = True
                self.retarget.set_ground_offset(self._ground_z_est)
            return

        dt = max(now - self._prev_ground_time, 1e-3)
        v_left = (left_z - self._prev_left_z) / dt
        v_right = (right_z - self._prev_right_z) / dt
        self._prev_ground_time = now
        self._prev_left_z = left_z
        self._prev_right_z = right_z

        if not self._ground_initialized:
            self._ground_init_samples.append(min_z)
            if len(self._ground_init_samples) >= self.args.ground_init_frames:
                self._ground_z_est = float(np.median(self._ground_init_samples))
                self._ground_initialized = True
                self.retarget.set_ground_offset(self._ground_z_est)
            return

        height_ok = (
            (left_z - self._ground_z_est) < self.args.ground_contact_height
            and (right_z - self._ground_z_est) < self.args.ground_contact_height
        )
        velocity_ok = (
            abs(v_left) < self.args.ground_contact_vel
            and abs(v_right) < self.args.ground_contact_vel
        )
        if height_ok and velocity_ok:
            alpha = self.args.ground_smooth
            self._ground_z_est = (1.0 - alpha) * self._ground_z_est + alpha * min_z
            self.retarget.set_ground_offset(self._ground_z_est)
        
    def update_visualization(self, qpos, smplx_data, viewer):
        """Update MuJoCo visualization"""
        if qpos is None:
            return
            
        # Clean custom geometry
        if hasattr(viewer, 'user_scn') and viewer.user_scn is not None:
            viewer.user_scn.ngeom = 0
            
        # Draw the task targets for reference
        if (
            smplx_data is not None
            and self.retarget is not None
            and not self.args.disable_draw_targets
        ):
            for robot_link, ik_data in self.retarget.ik_match_table1.items():
                body_name = ik_data[0]
                if body_name not in smplx_data:
                    continue
                draw_frame(
                    self.retarget.scaled_human_data[body_name][0] - self.retarget.ground,
                    R.from_quat(smplx_data[body_name][1]).as_matrix(),
                    viewer,
                    0.1,
                    orientation_correction=R.from_quat(ik_data[-1]),
                )
                
        # Update the simulation
        if qpos is not None:
            self.data.qpos[:] = qpos.copy()
            mj.mj_forward(self.model, self.data)
            
            # Camera follow the pelvis
            self._update_camera_position(viewer)
        
    def _update_camera_position(self, viewer):
        """Update camera to follow the robot"""
        FOLLOW_CAMERA = True
        if FOLLOW_CAMERA:
            robot_base_pos = self.data.xpos[self.model.body(self.robot_base).id]
            viewer.cam.lookat = robot_base_pos
            viewer.cam.distance = 3.0
    
    def determine_qpos_to_send(self, current_qpos):
        """Determine which qpos to publish based on current state"""
        current_state = self.state_machine.get_current_state()

        if current_state == "teleop":
            if current_qpos is not None:
                return current_qpos
            if self.last_qpos is not None:
                return self.last_qpos
        elif current_state == "pause":
            if self.last_qpos is not None:
                return self.last_qpos

        if self.default_qpos is not None:
            return self.default_qpos
        return current_qpos

    def send_body_to_lcm(self, qpos):
        """Publish retargeted robot data to LCM"""
        payload = self._build_lcm_payload(qpos)
        if payload is None:
            return
        dof_pos, root_pos, root_rot = payload
        timestamp_us = int(time.time() * 1e6)
        self.lcm_pub.publish_q(dof_pos, root_pos, root_rot, timestamp_us)
            
    def record_video_frame(self, viewer):
        """Record current frame to video if recording is enabled"""
        if not self.args.record_video or self.renderer is None:
            return
            
        self.renderer.update_scene(self.data, camera=viewer.cam)
        pixels = self.renderer.render()
        
        # Convert from RGB to BGR (OpenCV uses BGR)
        frame = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        self.video_writer.write(frame)
        
    def handle_exit_sequence(self, viewer=None):
        """Handle graceful exit."""
        qpos_to_send = self.determine_qpos_to_send(None)
        self.send_body_to_lcm(qpos_to_send)
        if viewer is not None:
            viewer.sync()
        self.rate.sleep()
                


    def initialize_all_systems(self):
        """Initialize all required systems"""
        print("Initializing teleop systems...")
        self.setup_teleop_data_streamer()
        self.setup_retargeting_system()
        self.setup_lcm_publisher()
        self.setup_gripper_publisher()
        self._build_lcm_qpos_indices()
        self.setup_mujoco_simulation()
        self.setup_video_recording()
        self.setup_rate_limiter()
        self.setup_recording()

        print("Teleop state machine initialized. Controls:")
        print("- Right controller key_one (A): Start/stop motion recording")
        print("- Keyboard: 's' start/stop recording, 'f' discard (if sshkeyboard is available)")
        print("- Left controller key_one (X): Exit program")
        print("- Left controller axis_click: Emergency stop - kills sim2real.sh process")
        print("- Left controller axis: Control root xy velocity")
        print("- Right controller axis: Control yaw velocity")
        print("- Left controller trigger: LEFT dex3 grip amount (0=open, 1=closed)")
        print("- Right controller trigger: RIGHT dex3 grip amount (0=open, 1=closed)")
        print("- Publishes retargeted body pose via LCM")
        print("- Auto-transition: idle -> teleop when motion data is available")
        print(f"Starting in state: {self.state_machine.get_current_state()}")

        if self.fps_monitor.enable_detailed_stats:
            print(f"- FPS measurement: ENABLED (detailed stats every {self.fps_monitor.detailed_print_interval} steps)")
        else:
            print(f"- FPS measurement: Quick stats only (every {self.fps_monitor.quick_print_interval} steps)")

        print("Ready to receive teleop data.")

    def run(self):
        """Main execution loop"""
        self.initialize_all_systems()
        
        def loop_body(viewer=None):
            """Single iteration of teleop loop."""
            # Get current teleop data
            smplx_data, _, _, controller_data, _ = self.get_teleop_data()

            # Update state machine
            if controller_data is not None:
                self.state_machine.update(controller_data)
            self._update_gripper_state(controller_data)
            if self.state_machine.consume_record_toggle():
                if self.record_enabled:
                    self.record_toggle = True
                else:
                    print("[record] Recording not enabled. Use --record_dir to enable.")

            # Check if we should exit
            if self.state_machine.should_exit():
                print("Exit requested via controller")
                self.handle_exit_sequence(viewer)
                return False

            # Process retargeting if we have data
            qpos = None
            if smplx_data is not None:
                qpos = self.process_retargeting(smplx_data)
                if viewer is not None:
                    self.update_visualization(qpos, smplx_data, viewer)
                self.state_machine.auto_transition(True)
            else:
                self.state_machine.auto_transition(False)

            qpos_to_send = self.determine_qpos_to_send(qpos)
            payload = self._build_lcm_payload(qpos_to_send)
            if payload is not None:
                dof_pos, root_pos, root_rot = payload
                timestamp_us = int(time.time() * 1e6)
                self.lcm_pub.publish_q(dof_pos, root_pos, root_rot, timestamp_us)
                if self.record_enabled and self.record_running:
                    self.record_buffer.append(
                        list(dof_pos) + list(root_pos) + list(root_rot) + [int(timestamp_us)]
                    )
                    if self.args.record_verbose:
                        print(f"[record] Recorded frame {self.record_buffer[-1]}")

            if self.record_enabled:
                if self.record_toggle and self.record_running:
                    if self.record_failed:
                        print("[record] Discarded (marked failed).")
                        self.record_buffer = []
                        self.record_failed = False
                    else:
                        print("[record] Save motion.")
                        self._save_record()
                    self.record_toggle = False
                    self.record_running = False
                elif self.record_toggle and not self.record_running:
                    print("[record] Start recording.")
                    self.record_buffer = []
                    self.record_toggle = False
                    self.record_running = True
                    self.record_saved = False

            # Update visualization and record video
            if viewer is not None:
                viewer.sync()
                self.record_video_frame(viewer)

            # FPS monitoring
            self.fps_monitor.tick()

            self.rate.sleep()
            return True

        try:
            if self.headless:
                print("Running in headless mode (no viewer/rendering) for higher FPS")
                while loop_body(viewer=None):
                    pass
            else:
                with mjv.launch_passive(
                    model=self.model,
                    data=self.data,
                    show_left_ui=False,
                    show_right_ui=False,
                ) as viewer:
                    viewer.opt.flags[mj.mjtVisFlag.mjVIS_TRANSPARENT] = 1

                    while viewer.is_running() and loop_body(viewer):
                        pass
        except KeyboardInterrupt:
            print("\nStopped teleop loop")
        finally:
            if self.record_enabled and self.record_running and not self.record_failed:
                self._save_record()
            if self.stop_listen is not None:
                try:
                    self.stop_listen()
                except Exception:
                    pass

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--robot",
        choices=["unitree_g1", "unitree_g1_with_hands"],
        default="unitree_g1",
    )
    parser.add_argument(
        "--record_video",
        action="store_true",
        help="Whether to record the video.",
    )
    parser.add_argument(
        "--lcm_channel",
        type=str,
        default="camera_reference_data",
        help="LCM channel name for robot control.",
    )
    parser.add_argument(
        "--actual_human_height",
        type=float,
        default=1.5,
        help="Actual human height for retargeting.",
    )   
    parser.add_argument(
        "--auto_ground",
        action="store_true",
        help="Update ground offset only when both feet are near ground.",
    )
    parser.add_argument(
        "--ground_init_frames",
        type=int,
        default=20,
        help="Frames to bootstrap ground estimate.",
    )
    parser.add_argument(
        "--ground_contact_height",
        type=float,
        default=0.02,
        help="Contact height threshold in meters.",
    )
    parser.add_argument(
        "--ground_contact_vel",
        type=float,
        default=0.15,
        help="Contact vertical velocity threshold in m/s.",
    )
    parser.add_argument(
        "--ground_smooth",
        type=float,
        default=0.1,
        help="Ground estimate smoothing factor.",
    )
    parser.add_argument(
        "--offset_to_ground",
        action="store_true",
        help="Force per-frame grounding (disables jump).",
    )
    parser.add_argument(
        "--target_fps",
        type=int,
        default=100,
        help="Target FPS for the teleop system.",
    )
    parser.add_argument(
        "--measure_fps",
        type=int,
        default=0,
        help="Measure and print detailed FPS statistics (0=disabled, 1=enabled).",
    )
    parser.add_argument(
        "--record_dir",
        type=str,
        default=None,
        help="Directory to save recorded robot motions (.npz).",
    )
    parser.add_argument(
        "--record_prefix",
        type=str,
        default="motion",
        help="Filename prefix for recorded motions.",
    )
    parser.add_argument(
        "--record_auto_start",
        action="store_true",
        help="Start recording immediately (otherwise press 's' to toggle).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without MuJoCo viewer/rendering to reduce overhead.",
    )
    parser.add_argument(
        "--disable_draw_targets",
        action="store_true",
        help="Skip drawing IK target frames to speed up rendering.",
    )
    parser.add_argument(
        "--record_verbose",
        action="store_true",
        help="Print every recorded frame (off by default for performance).",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    teleop_robot = XRobotTeleopToRobot(args)
    teleop_robot.run()
