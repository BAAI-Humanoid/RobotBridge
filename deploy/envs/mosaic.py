from typing import Dict, List, Optional, Union

import numpy as np
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from envs.base_env import BaseEnv

from utils.dataset import MosaicModelMeta, MotionDataset
from utils.dof import DoFAdapter
from utils.transformation import matrix_from_quat, subtract_frame_transforms, quat_rotate_inverse

from utils.data_pub import DataPublisher

import collections

info_pub = DataPublisher()
import os
from pathlib import Path

import imageio
import mujoco

class MosaicEnv(BaseEnv):
    """Environment wrapper that mimics RoboJuDo's Mosaic pipeline on top of our framework."""

    def __init__(self, config: DictConfig):
        super().__init__(config)

        cfg_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else config
        self.policy_cfg: Dict[str, Union[float, bool]] = cfg_dict.get("policy", {}) if isinstance(cfg_dict, dict) else {}
        self.motion_cfg: Dict[str, Union[float, bool, str]] = (
            cfg_dict.get("motion", {}) if isinstance(cfg_dict, dict) else {}
        )

        self.motion_loader = MotionDataset(self.motion_cfg, self.simulator)

        self.loop_motion = bool(self.motion_cfg.get("loop", False))
        self.playback_speed = float(self.motion_cfg.get("playback_speed", 1.0))
        self.max_timestep = int(self.policy_cfg.get("max_timestep", -1))
        self.without_state_estimator = bool(self.policy_cfg.get("without_state_estimator", True))
        self.action_beta = float(self.policy_cfg.get("action_beta", 1.0))

        self.policy_model_meta: Optional[MosaicModelMeta] = None
        self.policy_action_scales: Optional[np.ndarray] = None
        self.policy_default_joint_pos: Optional[np.ndarray] = None
        self.policy_joint_names: Optional[List[str]] = None
        self.policy_joint_index: Optional[Dict[str, int]] = None

        self._sim_to_policy: Optional[np.ndarray] = None
        self._policy_to_sim: Optional[np.ndarray] = None
        self._policy_dim: Optional[int] = None
        self._policy_to_sim_adapter: Optional[DoFAdapter] = None
        self._sim_to_policy_adapter: Optional[DoFAdapter] = None

        self.prev_policy_action: Optional[np.ndarray] = None
        self.time_step: float = 0.0
        self.motion_finished: bool = False

        self._pending_alignment_reset: bool = True

        self.last_ref_dof_pos = np.zeros(self.num_action, dtype=np.float64)

        # Teleop smoothing parameters
        self.teleop_pos_ema_alpha = float(self.policy_cfg.get("teleop_pos_ema_alpha", 0.3))
        self.teleop_vel_ema_alpha = float(self.policy_cfg.get("teleop_vel_ema_alpha", 0.5))
        self.teleop_pos_buffer_size = int(self.policy_cfg.get("teleop_pos_buffer_size", 5))

        # Buffers for smoothing
        self.teleop_pos_buffer = None  # Will be initialized in reset
        self.teleop_pos_smoothed = None
        self.teleop_vel_smoothed = None
        self.teleop_buffer_initialized = False

        self.history_length = int(self.policy_cfg.get("history_length", 1))
        self.obs_command_buffer = collections.deque([np.zeros(58) for _ in range(self.history_length)], maxlen=self.history_length)
        self.obs_motion_anchor_ori_b_buffer = collections.deque([np.zeros(6) for _ in range(self.history_length)], maxlen=self.history_length)
        self.obs_base_ang_vel_buffer = collections.deque([np.zeros(3) for _ in range(self.history_length)], maxlen=self.history_length)
        self.obs_joint_pos_rel_buffer = collections.deque([np.zeros(29) for _ in range(self.history_length)], maxlen=self.history_length)
        self.obs_joint_vel_rel_buffer = collections.deque([np.zeros(29) for _ in range(self.history_length)], maxlen=self.history_length)
        self.obs_prev_policy_action_buffer = collections.deque([np.zeros(29) for _ in range(self.history_length)], maxlen=self.history_length)
        self.obs_projected_gravity_buffer = collections.deque([np.zeros(3) for _ in range(self.history_length)], maxlen=self.history_length)
        
        self.gravity_w = np.zeros(3)
        self.gravity_w[2] = -1.0
        self.use_estimator = self.policy_cfg.get("use_estimator", False)

        self.hard_reset = False

        # Evaluation
        self.eval_mode = self.policy_cfg.get("eval_mode", False)
        ckpt_name = Path(self.policy_cfg.get("checkpoint")).stem 
        csv_name = f"metrics_{ckpt_name}.csv"
        log_dir = "logs" 
        os.makedirs(log_dir, exist_ok=True)
        self.metrics_path = os.path.join(log_dir, csv_name)
        
        self.motion_loader.set_metrics_file(self.metrics_path)
        logger.info(f"Metrics will be saved to: {self.metrics_path}")

        # save video
        self.init_video()

    def init_video(self):
        self.video_fps = 30
        self.dt = self.simulator.high_dt
        self.render_every = int(1 / (self.video_fps * self.dt))
        self.video_step_counter = 0
        self.video_episode_counter = 0
        self.frames = []
        
        self.offscreen_renderer = mujoco.Renderer(self.simulator.mujoco_model, height=640, width=480)

    # --------------------------------------------------------------------- #
    # Public API from agent
    # --------------------------------------------------------------------- #
    def configure_from_modelmeta(self, model_meta: MosaicModelMeta) -> None:
        """Apply joint configuration parsed from the Mosaic ONNX model."""
        logger.info("Applying joint configuration from ONNX model.")
        self.policy_model_meta = model_meta
        self.policy_joint_names = model_meta.joint_names
        self.policy_joint_index = model_meta.joint_index_map()
        self.policy_default_joint_pos = model_meta.default_joint_pos.astype(np.float32)
        self.policy_action_scales = model_meta.action_scale.astype(np.float32)
        self._policy_dim = len(self.policy_joint_names)

        self.prev_policy_action = np.zeros(self._policy_dim, dtype=np.float32)

        sim_joint_names = list(self.simulator.dof_names)
        if len(sim_joint_names) != self._policy_dim:
            raise RuntimeError(
                f"the number of policy joints ({self._policy_dim}) does not match the number of simulation joints ({len(sim_joint_names)}), please check the robot configuration."
            )

        reordered = model_meta.to_joint_order(sim_joint_names)
        sim_default = reordered["default_joint_pos"].astype(np.float32)
        sim_kps = reordered["joint_stiffness"].astype(np.float32)
        sim_kds = reordered["joint_damping"].astype(np.float32)

        self.simulator.default_angles = sim_default
        self.simulator.kps = sim_kps
        self.simulator.kds = sim_kds

        if hasattr(self.simulator.cfg, "asset"):
            asset_cfg = self.simulator.cfg.asset
            asset_cfg.default_angles = sim_default.tolist()
            asset_cfg.kps = sim_kps.tolist()
            asset_cfg.kds = sim_kds.tolist()

        sim_to_policy = np.array([self.policy_joint_index[name] for name in sim_joint_names], dtype=np.int32)
        policy_to_sim = np.zeros_like(sim_to_policy)
        policy_to_sim[sim_to_policy] = np.arange(self._policy_dim, dtype=np.int32)

        self._sim_to_policy = sim_to_policy
        self._policy_to_sim = policy_to_sim
        self.policy_default_joint_pos_sim = sim_default
        self.policy_action_scales_sim = self.policy_action_scales[sim_to_policy]
        self._policy_to_sim_adapter = DoFAdapter(self.policy_joint_names, sim_joint_names)
        self._sim_to_policy_adapter = DoFAdapter(sim_joint_names, self.policy_joint_names)

        logger.info("Mosaic policy metadata configured, {} degrees of freedom.", self._policy_dim)

    # --------------------------------------------------------------------- #
    # Overrides
    # --------------------------------------------------------------------- #
    def reset(self):
        if self.policy_model_meta is None:
            raise RuntimeError("MosaicEnv.reset() called before policy metadata was configured.")
        self.motion_loader.reset()
        self.motion_finished = False
        self.playback_speed = 1.0
        self.prev_policy_action = np.zeros(self._policy_dim, dtype=np.float32)

        # Reset teleop smoothing buffers
        self.teleop_buffer_initialized = False
        self.teleop_pos_buffer = None
        self.teleop_pos_smoothed = None
        self.teleop_vel_smoothed = None

        super().reset()
        if not self.eval_mode:
            self._interpolate_to_motion_start()
        self.compute_observation()

        if self.policy_cfg.get("save_video", False):
            self.check_save_video()

        return self.obs_buf_dict

    def step(self, action):
        if self.policy_model_meta is None:
            raise RuntimeError("MosaicEnv.step() called before policy metadata was configured.")

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != self._policy_dim:
            raise ValueError(f"Expected policy action of size {self._policy_dim}, received {action.size}.")

        smoothed = (1.0 - self.action_beta) * self.prev_policy_action + self.action_beta * action
        self.prev_policy_action = smoothed
        scaled = smoothed * self.policy_action_scales
        pd_target = scaled + self.policy_default_joint_pos
        sim_action = self._policy_vector_to_sim(pd_target)

        if self.policy_cfg.get("save_video", False):
            self.sample_video_frame()

        return super().step(sim_action[None, ...])

    def _get_command(self):
        command_data = self.motion_loader.get_data()
        return (
            command_data["command"],
            command_data["robot_anchor_pos_w"],
            command_data["robot_anchor_quat_w"],
            command_data["anchor_pos_w"],
            command_data["anchor_quat_w"],
        )
    
    def _get_command_teleop(self):
        self.simulator.get_state()
        raw_dof_pos = self.simulator.teleop_dof_pos
        info_pub.pub_vector("ref_dof_pos_raw", raw_dof_pos.tolist())

        # Apply smoothing using EMA and gradient-based velocity estimation
        ref_dof_pos, ref_dof_vel = self._smooth_teleop_data(raw_dof_pos)

        command = np.concatenate([ref_dof_pos, ref_dof_vel], axis=-1)

        # Info Publisher
        info_pub.pub_vector("ref_dof_pos", ref_dof_pos.tolist())
        info_pub.pub_vector("ref_dof_vel", ref_dof_vel.tolist())
        info_pub.step_publisher(0)

        robot_anchor_quat_w = self.simulator.torso_quat
        assert robot_anchor_quat_w is not None
        robot_anchor_pos_w = np.zeros(3, dtype=np.float64)
        anchor_pos_w = np.zeros(3, dtype=np.float64)
        anchor_quat_w = self.simulator.teleop_quat

        return (
            command,
            robot_anchor_pos_w,
            robot_anchor_quat_w,
            anchor_pos_w,
            anchor_quat_w,
        )

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

    # def _get_reference_markers_world(self) -> Optional[np.ndarray]:
    #     """Build reference markers (Nx3) in simulator robot frame for visualization."""
    #     command_data = self.motion_loader.get_data()
    #     body_pos_aligned = command_data.get("body_pos_w_aligned", None)
    #     if body_pos_aligned is None:
    #         return None
    #     body_pos_aligned = np.asarray(body_pos_aligned, dtype=np.float32).reshape(-1, 3)
    #     if body_pos_aligned.size == 0:
    #         return None

    #     # Convert reference bodies into an anchor-relative cloud first, to avoid "floating in the air".
    #     # (body_pos_aligned is in the same aligned frame as anchor_pos_w)
    #     ref_anchor_pos_aligned = np.asarray(command_data.get("anchor_pos_w", np.zeros(3)), dtype=np.float32).reshape(-1)[:3]
    #     body_rel = body_pos_aligned - ref_anchor_pos_aligned.reshape(1, 3)

    #     # Place around current robot anchor pose (torso/pelvis) with yaw only.
    #     robot_anchor_pos_w = np.asarray(command_data.get("robot_anchor_pos_w", self.simulator.root_trans_world), dtype=np.float32).reshape(-1)[:3]
    #     robot_anchor_quat_w = np.asarray(command_data.get("robot_anchor_quat_w", self.simulator.root_quat_world), dtype=np.float32).reshape(-1)[:4]  # xyzw
    #     try:
    #         yaw = float(sRot.from_quat(robot_anchor_quat_w).as_euler("xyz", degrees=False)[2])
    #     except Exception:
    #         yaw = 0.0
            
    #     yaw_rot = sRot.from_euler("z", yaw, degrees=False)
    #     markers_world = yaw_rot.apply(body_rel) + robot_anchor_pos_w.reshape(1, 3)
    #     return markers_world.astype(np.float32)

    def _get_reference_markers_world(self) -> Optional[np.ndarray]:
        """Build reference markers (Nx3) in simulator world frame for visualization."""
        if getattr(self.cfg.control, "use_teleop", False):
            body_pos_aligned = self.simulator.teleop_body_pos_w_aligned
        else:
            command_data = self.motion_loader.get_data()
            body_pos_aligned = command_data.get("body_pos_w_aligned", None)
            if body_pos_aligned is None:
                return None
            body_pos_aligned = np.asarray(body_pos_aligned, dtype=np.float32).reshape(-1, 3)

        if body_pos_aligned.size == 0:
            return None

        return body_pos_aligned.astype(np.float32)
    
    def compute_observation(self):
        BaseEnv._update_obs(self)
        
        if getattr(self.cfg.control, "use_teleop", False):
            command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w = self._get_command_teleop()
        else:
            command, robot_anchor_pos_w, robot_anchor_quat_w, anchor_pos_w, anchor_quat_w = self._get_command()
            
        pos, ori = subtract_frame_transforms(
            np.asarray(robot_anchor_pos_w, dtype=np.float32),
            np.asarray(robot_anchor_quat_w, dtype=np.float32),
            np.asarray(anchor_pos_w, dtype=np.float32),
            np.asarray(anchor_quat_w, dtype=np.float32),
        )
        if self.eval_mode:
            self.motion_loader._update_metrics()

        obs_ref_project_gravity = quat_rotate_inverse(np.roll(anchor_quat_w, 1), self.gravity_w)

        mat = matrix_from_quat(ori)

        obs_command = command
        obs_motion_anchor_pos_b = pos
        obs_motion_anchor_ori_b = mat[:, :2].flatten()

        obs_base_lin_vel = self.base_lin_vel.squeeze()
        obs_base_ang_vel = self.base_ang_vel.squeeze()
        obs_joint_pos_rel = self._sim_vector_to_policy(self.dof_pos.squeeze()) - self.policy_default_joint_pos
        obs_joint_vel_rel = self._sim_vector_to_policy(self.dof_vel.squeeze())
        obs_prev_policy_action = self.prev_policy_action

        self.obs_command_buffer.append(obs_command)
        self.obs_motion_anchor_ori_b_buffer.append(obs_motion_anchor_ori_b)
        self.obs_base_ang_vel_buffer.append(obs_base_ang_vel)
        self.obs_joint_pos_rel_buffer.append(obs_joint_pos_rel)
        self.obs_joint_vel_rel_buffer.append(obs_joint_vel_rel)
        self.obs_prev_policy_action_buffer.append(obs_prev_policy_action)

        obs_prop = np.concatenate([
            np.array(self.obs_command_buffer).reshape(1, -1),
            np.array(self.obs_motion_anchor_ori_b_buffer).reshape(1, -1),
            np.array(self.obs_base_ang_vel_buffer).reshape(1, -1),
            np.array(self.obs_joint_pos_rel_buffer).reshape(1, -1),
            np.array(self.obs_joint_vel_rel_buffer).reshape(1, -1),
            np.array(self.obs_prev_policy_action_buffer).reshape(1, -1),
        ], axis=1)

        if self.use_estimator:
            self.obs_projected_gravity_buffer.append(obs_ref_project_gravity)
            obs_estimator = np.concatenate([
                np.array(self.obs_command_buffer).reshape(1, -1),
                np.array(self.obs_projected_gravity_buffer).reshape(1, -1),
            ], axis=1)
            obs_prop = np.concatenate([obs_prop, obs_estimator], axis=1)

        self.obs_buf_dict = {
            "obs": obs_prop,
        }

    def _post_physics_step(self):
        super()._post_physics_step()
        self.motion_loader.post_step_callback()
        self.time_step += 1 * self.playback_speed
        if self.max_timestep > 0 and self.policy_time_step >= self.max_timestep:
            self.motion_finished = True
            self.playback_speed = 0.0

    def _physics_step(self):
        super()._physics_step()
        # Update visualization markers if simulator supports it.
        if getattr(self.simulator, "marker", False):
            markers_world = self._get_reference_markers_world()
            if markers_world is not None:
                self.simulator.update_marker_pos(markers_world[None, ...])

    # --------------------------------------------------------------------- #
    # Helpers
    # --------------------------------------------------------------------- #
    def _policy_vector_to_sim(self, policy_vector: np.ndarray) -> np.ndarray:
        if self._policy_to_sim_adapter is not None:
            template = np.zeros_like(self.policy_default_joint_pos_sim, dtype=np.float32)
            return self._policy_to_sim_adapter.fit(policy_vector, template=template)
        return policy_vector[self._sim_to_policy].astype(np.float32)

    def _sim_vector_to_policy(self, sim_vector: np.ndarray) -> np.ndarray:
        if self._sim_to_policy_adapter is not None:
            template = np.zeros(self._policy_dim, dtype=np.float32)
            return self._sim_to_policy_adapter.fit(sim_vector, template=template)
        result = np.zeros(self._policy_dim, dtype=np.float32)
        result[self._sim_to_policy] = sim_vector.astype(np.float32)
        return result

    @staticmethod
    def _wxyz_to_xyzw(quat_wxyz: np.ndarray) -> np.ndarray:
        return np.asarray([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Interpolation helper
    # ------------------------------------------------------------------ #
    def _interpolate_to_motion_start(self):
        """before the motion starts, interpolate the policy joints to the motion start"""
        steps = int(getattr(self.motion_loader, "interp_steps", 0))
        if steps <= 0:
            return

        target_policy = self.motion_loader.joint_pos  # motion 首帧（policy 关节顺序）
        target_sim = self._policy_vector_to_sim(target_policy)

        # Current simulation joint angles
        current_sim = np.asarray(self.simulator.dof_pos).squeeze().astype(np.float32)

        # Linear interpolation and advance physics
        for i in range(steps):
            alpha = float(i + 1) / float(steps)
            blended = (1.0 - alpha) * current_sim + alpha * target_sim
            self.simulator.apply_action(blended[None, ...])

        # Update internal state cache
        self.simulator.get_state()

    def next_motion(self, fail: bool = False):
        self.check_save_video()
        self.motion_loader.next_motion(fail)
        return self.reset()
    
    def _check_termination(self):
        self.hard_reset = self.simulator.check_termination()
        if self.hard_reset:
            if self.eval_mode:
                self.next_motion(fail=True)
            self._save_collected_traj(False)
            self._reset_envs(True)
            self.compute_observation()
        self.hard_reset = False

    def check_save_video(self):
        if len(self.frames) > 0 and not self.hard_reset:
            self.save_video()

        self.frames = []
        self.video_step_counter = 0
        self.video_episode_counter += 1

    def save_video(self):
        filename = f"/home/ws/hbs/RobotBridge/videos_offline/episode_{self.video_episode_counter}.mp4"
        print(f"Saving video: {filename}")
        imageio.mimsave(filename, self.frames, fps=self.video_fps)
        self.frames = []

    def sample_video_frame(self):
        if self.video_step_counter % self.render_every == 0:

            self.offscreen_renderer.update_scene(self.simulator.mujoco_data, camera=self.simulator.viewer.cam) # 或者用 ID
            pixels = self.offscreen_renderer.render()
            self.frames.append(pixels)

        self.video_step_counter += 1