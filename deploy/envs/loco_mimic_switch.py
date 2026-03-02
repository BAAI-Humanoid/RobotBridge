"""
Unified environment that supports switching between locomotion and mimic policies.
Handles different DoF requirements (12 for locomotion, 29 for mimic).
"""

import collections
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from scipy.spatial.transform import Rotation as sRot

from envs.base_env import BaseEnv
from utils.dataset import MosaicModelMeta, MotionDataset
from utils.teleop import PygameKeyboardTeleop, RealStickTeleop, TerminalKeyboardTeleop
from utils.dof import DoFAdapter
from utils.transformation import quat_rotate_inverse, subtract_frame_transforms, matrix_from_quat


class LocoMimicSwitchEnv(BaseEnv):
    """
    Unified environment that supports switching between locomotion (12 DoF) and mimic (29 DoF) policies.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)

        cfg_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
        
        # Locomotion configuration
        self.loco_cfg = cfg_dict.get("locomotion", {}) if isinstance(cfg_dict, dict) else {}
        self.teleop_cfg = cfg_dict.get("teleop", {}) if isinstance(cfg_dict, dict) else {}
        self.loco_policy_control = cfg_dict.get("locomotion_policy_control", {}) if isinstance(cfg_dict, dict) else {}
        
        # Mimic configuration
        self.mimic_policy_cfg = cfg_dict.get("mimic", {}) if isinstance(cfg_dict, dict) else {}
        self.motion_cfg = cfg_dict.get("motion", {}) if isinstance(cfg_dict, dict) else {}
        
        # Policy state
        self.policy_mode = "locomotion"  # "locomotion" or "mimic"
        self.pending_policy_switch = None  # Pending policy switch request from teleop
        self.override_dof_pos = None  # Override DoF positions during interpolation
        
        # Initialize locomotion components
        self._init_locomotion()
        
        # Initialize mimic components
        self._init_mimic()
        
        # Override DoF configuration
        self.upper_dof_num = int(cfg_dict.get("upper_dof_num", 17)) if isinstance(cfg_dict, dict) else 17
        self.lower_dof_num = self.simulator.num_action - self.upper_dof_num
        
        logger.info(
            "LocoMimicSwitchEnv initialized | Total DoF: {}, Lower: {}, Upper: {}",
            self.simulator.num_action,
            self.lower_dof_num,
            self.upper_dof_num,
        )

    def _init_locomotion(self):
        """Initialize locomotion-specific components."""
        # LEVEL locomotion parameters
        self.cmd_scale = np.asarray(self.loco_cfg.get("cmd_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        self.height_cmd = float(self.loco_cfg.get("height_cmd", 0.75))
        self.ang_vel_scale = float(self.loco_cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale = float(self.loco_cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(self.loco_cfg.get("dof_vel_scale", 0.05))
        self.gait_phase_init = np.asarray(self.loco_cfg.get("gait_phase_init", [0.38, 0.38]), dtype=np.float32)
        self.gait_period = float(self.loco_cfg.get("gait_period", 0.8))
        self.stand_threshold = float(self.loco_cfg.get("stand_threshold", 0.1))
        self.obs_history_len = int(self.loco_cfg.get("obs_history_len", 10))
        self.num_obs_single_cfg = int(self.loco_cfg.get("num_obs_single", 48))
        self.action_obs = np.zeros((1, self.simulator.num_action), dtype=np.float32)

        # Runtime state for locomotion
        self.gait_phase = self.gait_phase_init.copy()
        self.flag_stand = True
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        self.loco_obs = np.zeros(self.num_obs_single_cfg * self.obs_history_len, dtype=np.float32)
        
        # Command velocity
        self.command_lin_vel = np.zeros((1, 3), dtype=np.float32)
        self.command_ang_vel = np.zeros((1, 3), dtype=np.float32)
        
        # Initialize teleop
        self._init_teleop()

    def _init_teleop(self):
        """Initialize teleoperation interface."""
        self._kb_teleop = None
        self._stick_teleop = None
        self._keyboard_backend = "auto"
        
        if getattr(self.simulator, "is_real", False):
            stick = self.teleop_cfg.get("real_stick", {}) if isinstance(self.teleop_cfg, dict) else {}
            self._stick_teleop = RealStickTeleop(
                vx_scale=float(stick.get("vx_scale", 0.6)),
                vy_scale=float(stick.get("vy_scale", 0.6)),
                yaw_scale=float(stick.get("yaw_scale", 0.6)),
                smoothing=float(stick.get("smoothing", 0.5)),
            )
        else:
            kb = self.teleop_cfg.get("keyboard", {}) if isinstance(self.teleop_cfg, dict) else {}
            self._keyboard_backend = str(kb.get("backend", "auto"))
            
            if self._keyboard_backend in ("auto", "pygame"):
                try:
                    cmd_screen = kb.get("command_screen_size", [360, 50])
                    keyboard_step = kb.get(
                        "pygame_step",
                        {"vx": 0.1, "vy": 0.1, "yaw": 0.1, "height": 0.1},
                    )
                    keyboard_limits = kb.get(
                        "pygame_limits",
                        {"vx": [-0.8, 1.2], "vy": [-0.6, 0.6], "yaw": [-0.8, 0.8], "height": [-0.5, 0.0]},
                    )
                    height_limits = kb.get("height_limits", [0.25, 0.75])
                    self._kb_teleop = PygameKeyboardTeleop(
                        command_screen_size=tuple(cmd_screen),
                        keyboard_step=(
                            float(keyboard_step["vx"]),
                            float(keyboard_step["vy"]),
                            float(keyboard_step["yaw"]),
                            float(keyboard_step["height"]),
                        ),
                        keyboard_limits=(
                            tuple(keyboard_limits["vx"]),
                            tuple(keyboard_limits["vy"]),
                            tuple(keyboard_limits["yaw"]),
                            tuple(keyboard_limits["height"]),
                        ),
                        base_height_cmd=float(self.height_cmd),
                        height_limits=tuple(height_limits),
                    )
                    self._keyboard_backend = "pygame"
                except Exception as exc:
                    if self._keyboard_backend == "pygame":
                        raise
                    logger.warning("pygame keyboard control unavailable, falling back to terminal: {}", exc)
                    self._keyboard_backend = "terminal"
            
            if self._kb_teleop is None:
                step = kb.get("step", {"vx": 0.1, "vy": 0.1, "yaw": 0.1})
                limits = kb.get(
                    "limits",
                    {"vx": [-0.8, 1.2], "vy": [-0.6, 0.6], "yaw": [-0.8, 0.8]},
                )
                self._kb_teleop = TerminalKeyboardTeleop(
                    step=(float(step["vx"]), float(step["vy"]), float(step["yaw"])),
                    limits=(tuple(limits["vx"]), tuple(limits["vy"]), tuple(limits["yaw"])),
                    print_help=bool(kb.get("print_help", True)),
                )
                self._keyboard_backend = "terminal"

    def _init_mimic(self):
        """Initialize mimic-specific components."""
        # Save original locomotion PD parameters
        self.loco_kps = self.simulator.kps.copy()
        self.loco_kds = self.simulator.kds.copy()
        logger.info("Saved locomotion PD parameters (kp/kd)")
        
        self.motion_loader = MotionDataset(self.motion_cfg, self.simulator)
        
        self.loop_motion = bool(self.motion_cfg.get("loop", False))
        self.playback_speed = float(self.motion_cfg.get("playback_speed", 1.0))
        self.max_timestep = int(self.mimic_policy_cfg.get("max_timestep", -1))
        self.without_state_estimator = bool(self.mimic_policy_cfg.get("without_state_estimator", True))
        self.action_beta_mimic = float(self.mimic_policy_cfg.get("action_beta", 1.0))

        # Mimic policy metadata (to be configured from ONNX)
        self.mimic_model_meta: Optional[MosaicModelMeta] = None
        self.mimic_joint_names: Optional[List[str]] = None
        self.mimic_default_joint_pos: Optional[np.ndarray] = None
        self.mimic_action_scales: Optional[np.ndarray] = None
        self._mimic_dim: Optional[int] = None
        self.mimic_kps: Optional[np.ndarray] = None  # Mimic policy kp parameters
        self.mimic_kds: Optional[np.ndarray] = None  # Mimic policy kd parameters
        
        self.prev_mimic_action: Optional[np.ndarray] = None
        self.time_step: float = 0.0
        self.motion_finished: bool = False
        
        self._sim_to_mimic_adapter: Optional[DoFAdapter] = None
        self._mimic_to_sim_adapter: Optional[DoFAdapter] = None

        self.history_length = int(self.mimic_policy_cfg.get("history_length", 1)) 
        self.eval_mode = self.mimic_policy_cfg.get("eval_mode", False)

        self.mimic_obs_bufs = {
            "command": collections.deque(maxlen=self.history_length),      # 58
            "anchor_ori": collections.deque(maxlen=self.history_length),   # 6
            "ang_vel": collections.deque(maxlen=self.history_length),      # 3
            "joint_pos": collections.deque(maxlen=self.history_length),    # 29
            "joint_vel": collections.deque(maxlen=self.history_length),    # 29
            "action": collections.deque(maxlen=self.history_length),       # 29
        }
        self._reset_mimic_obs_buffers()

        self.teleop_pos_ema_alpha = float(self.teleop_cfg.get("pos_ema_alpha", 0.3))
        self.teleop_buffer_initialized = False

    def configure_mimic_from_modelmeta(self, model_meta: MosaicModelMeta) -> None:
        """Configure mimic policy from ONNX model metadata."""
        logger.info("Configuring mimic policy from ONNX model metadata.")
        self.mimic_model_meta = model_meta
        self.mimic_joint_names = model_meta.joint_names
        self.mimic_default_joint_pos = model_meta.default_joint_pos.astype(np.float32)
        self.mimic_action_scales = model_meta.action_scale.astype(np.float32)
        self._mimic_dim = len(self.mimic_joint_names)
        
        self.prev_mimic_action = np.zeros(self._mimic_dim, dtype=np.float32)
        
        # Build adapters
        sim_joint_names = list(self.simulator.dof_names)
        self._sim_to_mimic_adapter = DoFAdapter(sim_joint_names, self.mimic_joint_names)
        self._mimic_to_sim_adapter = DoFAdapter(self.mimic_joint_names, sim_joint_names)
        
        # Extract and adapt mimic PD parameters to simulator DoF order
        mimic_kps_policy = model_meta.joint_stiffness.astype(np.float32)
        mimic_kds_policy = model_meta.joint_damping.astype(np.float32)
        
        # Convert from mimic policy joint order to simulator joint order
        # Create full-size arrays with locomotion defaults, then fill in mimic values
        self.mimic_kps = self.loco_kps.copy()
        self.mimic_kds = self.loco_kds.copy()
        
        # Map mimic parameters to simulator joints
        for mimic_idx, mimic_joint in enumerate(self.mimic_joint_names):
            if mimic_joint in sim_joint_names:
                sim_idx = sim_joint_names.index(mimic_joint)
                self.mimic_kps[sim_idx] = mimic_kps_policy[mimic_idx]
                self.mimic_kds[sim_idx] = mimic_kds_policy[mimic_idx]
        
        logger.info("Mimic policy configured with {} DoF", self._mimic_dim)
        logger.info("Mimic PD parameters extracted and adapted to simulator DoF order")

    def _reset_mimic_obs_buffers(self):
        """Initialize mimic observation history buffers."""
        command_horizon = self.motion_cfg.get("command_horizon", 1)
        dim_map = {"command": 58*command_horizon, "anchor_ori": 6, "ang_vel": 3, "joint_pos": 29, "joint_vel": 29, "action": 29}
        for key in self.mimic_obs_bufs:
            self.mimic_obs_bufs[key].clear()
            for _ in range(self.history_length):
                self.mimic_obs_bufs[key].append(np.zeros(dim_map[key], dtype=np.float32))

    def _smooth_teleop_data(self, raw_pos: np.ndarray) -> tuple:
        """
        Smooth teleop position and compute velocity using buffered gradient.

        Args:
            raw_pos: Raw teleop joint positions [num_action]

        Returns:
            tuple: (smoothed_pos, smoothed_vel)
        """
        # Initialize buffers on first call
        if not self.teleop_buffer_initialized:
            self.teleop_pos_buffer = np.tile(raw_pos, (self.teleop_pos_buffer_size, 1))
            self.teleop_pos_smoothed = raw_pos.copy()
            self.teleop_vel_smoothed = np.zeros_like(raw_pos)
            self.teleop_buffer_initialized = True
            return self.teleop_pos_smoothed, self.teleop_vel_smoothed

        # EMA smoothing for position
        smoothed_pos = (self.teleop_pos_ema_alpha * raw_pos +
                       (1.0 - self.teleop_pos_ema_alpha) * self.teleop_pos_smoothed)

        # Update position buffer (rolling window)
        self.teleop_pos_buffer = np.roll(self.teleop_pos_buffer, shift=-1, axis=0)
        self.teleop_pos_buffer[-1] = smoothed_pos

        # Compute velocity using gradient over buffer
        # Using central difference for middle points, forward/backward for edges
        dt = self.simulator.high_dt
        buffer_len = self.teleop_pos_buffer.shape[0]

        if buffer_len >= 3:
            # Use gradient over the buffer for more stable velocity estimation
            # Simple central difference: v = (pos[i+1] - pos[i-1]) / (2*dt)
            # For the latest velocity, use last 3 points
            vel_raw = (self.teleop_pos_buffer[-1] - self.teleop_pos_buffer[-3]) / (2.0 * dt)
        else:
            # Fallback to simple difference
            vel_raw = (smoothed_pos - self.teleop_pos_smoothed) / dt

        # # EMA smoothing for velocity
        # smoothed_vel = (self.teleop_vel_ema_alpha * vel_raw +
        #                (1.0 - self.teleop_vel_ema_alpha) * self.teleop_vel_smoothed)

        smoothed_vel = vel_raw

        # Update internal state
        self.teleop_pos_smoothed = smoothed_pos
        self.teleop_vel_smoothed = smoothed_vel

        return smoothed_pos, smoothed_vel

    def set_policy_mode(self, mode: str):
        """Switch between 'locomotion' and 'mimic' policy modes."""
        if mode not in ["locomotion", "mimic"]:
            raise ValueError(f"Invalid policy mode: {mode}")
        
        old_mode = self.policy_mode
        self.policy_mode = mode
        
        if old_mode != mode:
            logger.info("Policy mode switched from {} to {}", old_mode, mode)
            
            # Switch PD control parameters
            if mode == "locomotion":
                self._switch_to_locomotion_pd_params()
            elif mode == "mimic":
                self._switch_to_mimic_pd_params()
                self._reset_mimic_obs_buffers()
            
            # Immediately recompute observation for the new policy mode
            self.compute_observation()

    def set_command_velocity(self, vx: float, vy: float, yaw: float):
        """Set locomotion command velocity."""
        self.command_lin_vel[0, 0] = vx
        self.command_lin_vel[0, 1] = vy
        self.command_ang_vel[0, 0] = yaw

    def reset(self):
        """Reset environment based on current policy mode."""
        self._reset_locomotion()
        if self.mimic_model_meta is not None:
            self._reset_mimic()
        
        super().reset()
        return self.obs_buf_dict

    def _reset_locomotion(self):
        """Reset locomotion-specific state."""
        self.gait_phase = self.gait_phase_init.copy()
        self.flag_stand = True
        self.command_lin_vel *= 0
        self.command_ang_vel *= 0
        self.action_obs *= 0
        self.obs_history.clear()
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.num_obs_single_cfg, dtype=np.float32))

    def _reset_mimic(self):
        """Reset mimic-specific state."""
        self.motion_loader.reset()
        self.motion_finished = False
        self.playback_speed = 1.0
        self.prev_mimic_action = np.zeros(self._mimic_dim, dtype=np.float32)
        self.time_step = 0.0

    def compute_observation(self):
        """Compute observation based on current policy mode."""
        super()._update_obs()
        
        # Always update commands (keyboard/joystick) regardless of policy mode
        self._update_commands()
        
        if self.policy_mode == "locomotion":
            self._compute_locomotion_observation()
        elif self.policy_mode == "mimic":
            self._compute_mimic_observation()
        else:
            raise ValueError(f"Unknown policy mode: {self.policy_mode}")

    def _compute_locomotion_observation(self):
        """
        Compute LEVEL locomotion observation.
        Note: Commands (keyboard/joystick) are already updated in compute_observation().
        """
        cmd = np.array(
            [self.command_lin_vel[0, 0], self.command_lin_vel[0, 1], self.command_ang_vel[0, 0]],
            dtype=np.float32,
        )
        
        single_obs = self._build_single_locomotion_obs(cmd, self.height_cmd)
        self.obs_history.append(single_obs)
        
        # Concatenate history
        for i, hist_obs in enumerate(self.obs_history):
            start = i * self.num_obs_single_cfg
            self.loco_obs[start : start + self.num_obs_single_cfg] = hist_obs
        
        self.obs_buf_dict = {"actor_obs": self.loco_obs[None, ...]}
        
        # Update gait phase
        self._update_gait_phase(cmd)

    def _build_single_locomotion_obs(self, cmd: np.ndarray, height_cmd: float) -> np.ndarray:
        """Build single timestep locomotion observation (LEVEL format)."""
        # Only use lower body joints for locomotion
        qj = self.dof_pos.reshape(-1)[:self.lower_dof_num]
        dqj = self.dof_vel.reshape(-1)[:self.lower_dof_num]
        omega = self.base_ang_vel.reshape(-1)
        
        quat_xyzw = self.root_quat.reshape(-1)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        gravity_orientation = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        
        default_angles = self.simulator.default_angles[self.simulator.active_dof_idx][:self.lower_dof_num]
        
        qj_scaled = (qj - default_angles) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        omega_scaled = omega * self.ang_vel_scale
        
        self.flag_stand = np.linalg.norm(cmd[:3]) < self.stand_threshold
        gait_sin = np.sin(2 * np.pi * self.gait_phase)
        
        action = self.action_obs.reshape(-1)[:self.lower_dof_num]
        
        single_obs = np.zeros(self.num_obs_single_cfg, dtype=np.float32)
        single_obs[0:3] = cmd * self.cmd_scale
        single_obs[3] = height_cmd
        single_obs[4:7] = omega_scaled
        single_obs[7:10] = gravity_orientation
        single_obs[10 : 10 + self.lower_dof_num] = qj_scaled
        single_obs[10 + self.lower_dof_num : 10 + 2 * self.lower_dof_num] = dqj_scaled
        single_obs[10 + 2 * self.lower_dof_num : 10 + 3 * self.lower_dof_num] = action
        single_obs[10 + 2 * self.lower_dof_num + 12 : 12 + 3 * self.lower_dof_num] = gait_sin
        
        return single_obs

    def _update_gait_phase(self, cmd: np.ndarray):
        """Update gait phase for locomotion."""
        gait_dt = self.simulator.low_dt * self.simulator.decimation
        self.gait_phase = np.remainder(self.gait_phase + gait_dt * self.gait_period, 1.0).astype(np.float32)
        
        if self.flag_stand and np.any(np.abs(self.gait_phase - 0.38) < 0.05):
            self.gait_phase = np.array([0.38, 0.38], dtype=np.float32)
        elif (not self.flag_stand) and np.all(np.abs(self.gait_phase - 0.38) < 0.05):
            self.gait_phase = np.array([0.38, 0.88], dtype=np.float32)

    def _update_commands(self):
        """Update locomotion commands from teleop."""
        if getattr(self.simulator, "is_real", False):
            if self._stick_teleop is None:
                return
            cmd = self._stick_teleop.update_from_sim(self.simulator)
            self.set_command_velocity(cmd.vx, cmd.vy, cmd.yaw)
            self._handle_policy_switch(cmd)
            return
        
        if self._kb_teleop is None:
            return
        cmd = self._kb_teleop.update()
        self.set_command_velocity(cmd.vx, cmd.vy, cmd.yaw)
        self._handle_policy_switch(cmd)
        if hasattr(cmd, "height"):
            try:
                base_h = float(self.loco_cfg.get("height_cmd", self.height_cmd))
                height_limits = self.teleop_cfg.get("keyboard", {}).get("height_limits", [0.25, 0.75])
                self.height_cmd = float(np.clip(base_h + float(cmd.height), float(height_limits[0]), float(height_limits[1])))
            except Exception:
                pass
    
    def _handle_policy_switch(self, cmd):
        """Handle policy switching from teleop command."""
        if hasattr(cmd, "policy_switch") and cmd.policy_switch is not None:
            # Store the pending switch request
            # The agent's interpolation manager will handle the actual switching
            if cmd.policy_switch in ["locomotion", "mimic"]:
                if cmd.policy_switch != self.policy_mode:
                    logger.info(f"Policy switch requested: {self.policy_mode} -> {cmd.policy_switch}")
                    self.pending_policy_switch = cmd.policy_switch
    
    def get_pending_policy_switch(self):
        """Get and clear pending policy switch request."""
        switch = self.pending_policy_switch
        self.pending_policy_switch = None
        return switch
    
    def _switch_to_locomotion_pd_params(self):
        """Switch to locomotion PD control parameters."""
        if self.loco_kps is None or self.loco_kds is None:
            logger.warning("Locomotion PD parameters not initialized")
            return
        
        self.simulator.kps = self.loco_kps.copy()
        self.simulator.kds = self.loco_kds.copy()
        logger.info("Switched to locomotion PD parameters")
        logger.debug("Locomotion kp range: [{:.1f}, {:.1f}]", self.loco_kps.min(), self.loco_kps.max())
        logger.debug("Locomotion kd range: [{:.1f}, {:.1f}]", self.loco_kds.min(), self.loco_kds.max())
    
    def _switch_to_mimic_pd_params(self):
        """Switch to mimic PD control parameters."""
        if self.mimic_kps is None or self.mimic_kds is None:
            logger.warning("Mimic PD parameters not configured, using locomotion parameters")
            return
        
        self.simulator.kps = self.mimic_kps.copy()
        self.simulator.kds = self.mimic_kds.copy()
        logger.info("Switched to mimic PD parameters")
        logger.debug("Mimic kp range: [{:.1f}, {:.1f}]", self.mimic_kps.min(), self.mimic_kps.max())
        logger.debug("Mimic kd range: [{:.1f}, {:.1f}]", self.mimic_kds.min(), self.mimic_kds.max())

    def _compute_mimic_observation(self):
        """
        Compute Mosaic observation.
        Note: Commands (keyboard/joystick) are already updated in compute_observation().
        """
        if self.mimic_model_meta is None:
            raise RuntimeError("Mimic policy not configured")
        
        command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w = self._get_mimic_command()
        
        pos, ori = subtract_frame_transforms(
            np.asarray(robot_anchor_pos_w, dtype=np.float32),
            np.asarray(robot_anchor_quat_w, dtype=np.float32),
            np.asarray(anchor_pos_w, dtype=np.float32),
            np.asarray(anchor_quat_w, dtype=np.float32),
        )
        
        mat = matrix_from_quat(ori)
        
        # Convert sim DoF to mimic policy DoF
        curr_joint_pos_rel = self._sim_to_mimic(self.dof_pos.squeeze()) - self.mimic_default_joint_pos
        curr_joint_vel_rel = self._sim_to_mimic(self.dof_vel.squeeze())
        
        # Update History Deque
        self.mimic_obs_bufs["command"].append(command)
        self.mimic_obs_bufs["anchor_ori"].append(mat[:, :2].flatten())
        self.mimic_obs_bufs["ang_vel"].append(self.base_ang_vel.squeeze())
        self.mimic_obs_bufs["joint_pos"].append(curr_joint_pos_rel)
        self.mimic_obs_bufs["joint_vel"].append(curr_joint_vel_rel)
        self.mimic_obs_bufs["action"].append(self.prev_mimic_action)

        obs_prop = np.concatenate([
            np.array(self.mimic_obs_bufs["command"]).flatten(),
            np.array(self.mimic_obs_bufs["anchor_ori"]).flatten(),
            np.array(self.mimic_obs_bufs["ang_vel"]).flatten(),
            np.array(self.mimic_obs_bufs["joint_pos"]).flatten(),
            np.array(self.mimic_obs_bufs["joint_vel"]).flatten(),
            np.array(self.mimic_obs_bufs["action"]).flatten(),
        ])
        
        self.obs_buf_dict = {"obs": obs_prop[None, ...]}

    def _get_mimic_command(self):
        """Get mimic motion command data."""
        command_data = self.motion_loader.get_data()
        return (
            command_data["command"],
            command_data["robot_anchor_pos_w"],
            command_data["robot_anchor_quat_w"],
            command_data["anchor_pos_w"],
            command_data["anchor_quat_w"],
        )

    def _sim_to_mimic(self, sim_vector: np.ndarray) -> np.ndarray:
        """Convert simulator joint vector to mimic policy joint vector."""
        if self._sim_to_mimic_adapter is not None:
            template = np.zeros(self._mimic_dim, dtype=np.float32)
            return self._sim_to_mimic_adapter.fit(sim_vector, template=template)
        return sim_vector

    def _mimic_to_sim(self, mimic_vector: np.ndarray) -> np.ndarray:
        """Convert mimic policy joint vector to simulator joint vector."""
        if self._mimic_to_sim_adapter is not None:
            # Use simulator's default angles as template so unmatched joints keep default pose
            template = self.simulator.default_angles[self.simulator.active_dof_idx].copy()
            return self._mimic_to_sim_adapter.fit(mimic_vector, template=template)
        return mimic_vector

    def step(self, action):
        """Step environment with action from current policy."""
        if self.policy_mode == "locomotion":
            return self._step_locomotion(action)
        elif self.policy_mode == "mimic":
            return self._step_mimic(action)
        else:
            raise ValueError(f"Unknown policy mode: {self.policy_mode}")

    def _step_locomotion(self, action):
        """Step with locomotion action (12 DoF for lower body)."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        # clip actions
        clip_action_limit = self.loco_policy_control['action_clip_value']
        self.action_obs = np.clip(action, -clip_action_limit, clip_action_limit)

        # sclaing the actions
        lowerbody_default_angles = self.simulator.default_angles[self.simulator.active_dof_idx][:self.lower_dof_num]
        tgt_dof_pos = lowerbody_default_angles + self.action_obs * self.loco_policy_control['action_scale']
        
        # Create full DoF action: locomotion controls lower body, upper body stays at default or override
        full_action = np.zeros((1, self.simulator.num_action), dtype=np.float32)
        full_action[0, :self.lower_dof_num] = tgt_dof_pos[:self.lower_dof_num]
        
        # Upper body: use override positions during interpolation, otherwise use default
        if self.override_dof_pos is not None:
            full_action[0, self.lower_dof_num:] = self.override_dof_pos[self.lower_dof_num:]
        else:
            full_action[0, self.lower_dof_num:] = self.simulator.default_angles[self.simulator.active_dof_idx][self.lower_dof_num:]
        
        return super().step(full_action)
    
    def _step_mimic(self, action):
        """Step with mimic action (29 DoF for full body)."""
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        
        if action.size != self._mimic_dim:
            raise ValueError(f"Expected mimic action of size {self._mimic_dim}, received {action.size}.")
        
        # Apply action smoothing
        smoothed = (1.0 - self.action_beta_mimic) * self.prev_mimic_action + self.action_beta_mimic * action
        self.prev_mimic_action = smoothed
        
        # Scale and add default position
        scaled = smoothed * self.mimic_action_scales
        pd_target = scaled + self.mimic_default_joint_pos
        
        # Convert to simulator joint order
        sim_action = self._mimic_to_sim(pd_target)
        
        return super().step(sim_action[None, ...])

    def _post_physics_step(self):
        """Post-step processing based on policy mode."""
        super()._post_physics_step()
        
        if self.policy_mode == "mimic":
            self.motion_loader.post_step_callback()
            self.time_step += 1 * self.playback_speed
            if self.max_timestep > 0 and self.time_step >= self.max_timestep:
                self.motion_finished = True
                self.playback_speed = 0.0

