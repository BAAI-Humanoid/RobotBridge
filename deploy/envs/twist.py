import os
import sys
from collections import deque
from typing import Dict, Optional, Sequence

import mujoco
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from simulator.mujoco import Mujoco
from utils.dataset import MotionDataset
from utils.motion_lib.rotations import quat_rotate_inverse

ANKLE_IDX = [4, 5, 10, 11]
ACTION_SCALE = 0.5
WRIST_IDS_25 = [19, 24]

TWIST_G1_25_DOF_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint"
]

TWIST_G1_23_DOF_NAMES = [n for n in TWIST_G1_25_DOF_NAMES if "wrist" not in n]

def quat_to_euler_wxyz_twist(q):
    qw, qx, qy, qz = q
    roll = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
    yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
    return np.array([roll, pitch, yaw], dtype=np.float32)

def _resolve_path(path: Optional[str]) -> Optional[str]:
    if not path:
        return None
    expanded = os.path.expanduser(path)
    if os.path.isabs(expanded):
        return expanded
    return os.path.abspath(os.path.join(project_root, expanded))

def _select_by_joint_names(
    values: Sequence[float],
    joint_order: Dict[str, int],
    target_names: Sequence[str],
    label: str,
) -> list:
    if values is None:
        raise ValueError(f"Missing {label} in asset config.")
    values_list = list(values)
    if not joint_order:
        if len(values_list) != len(target_names):
            raise ValueError(
                f"{label} length {len(values_list)} does not match expected {len(target_names)}."
            )
        return [float(v) for v in values_list]

    missing = [name for name in target_names if name not in joint_order]
    if missing:
        raise ValueError(f"{label} missing joint names: {missing}")
    return [float(values_list[joint_order[name]]) for name in target_names]

def _joint_indices(
    joint_order: Dict[str, int],
    target_names: Sequence[str],
    label: str,
) -> list:
    if not joint_order:
        raise ValueError(f"{label} requires asset.joint_order.")
    missing = [name for name in target_names if name not in joint_order]
    if missing:
        raise ValueError(f"{label} missing joint names: {missing}")
    return [int(joint_order[name]) for name in target_names]

def _map_active_indices(
    active_idx: Sequence[int],
    target_idx: Sequence[int],
    label: str,
) -> list:
    active_map = {int(idx): i for i, idx in enumerate(active_idx)}
    missing = [idx for idx in target_idx if idx not in active_map]
    if missing:
        raise ValueError(f"{label} indices not active: {missing}")
    return [active_map[idx] for idx in target_idx]

class TwistEnv:
    def __init__(self, config: DictConfig, asset_cfg: Optional[DictConfig] = None):
        self.config = config

        cfg_dict = OmegaConf.to_container(config, resolve=True) if isinstance(config, DictConfig) else (config or {})
        self.device = cfg_dict.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_cfg = cfg_dict.get("policy", {}) or {}
        self.motion_cfg = cfg_dict.get("motion", {}) or {}
        self.control_cfg = cfg_dict.get("control", {}) or {}
        self.robot_cfg = cfg_dict.get("robot", {}) or {}

        asset_cfg = cfg_dict.get("asset", {}) or {}
        asset_dict = (
            OmegaConf.to_container(asset_cfg, resolve=True)
            if isinstance(asset_cfg, DictConfig)
            else dict(asset_cfg)
        )
        self.asset_cfg = OmegaConf.create(asset_dict)

        joint_order = dict(asset_dict.get("joint_order", {}))
        self.body_indices = _joint_indices(joint_order, TWIST_G1_23_DOF_NAMES, "body joint indices")
        self.wrist_indices = _joint_indices(
            joint_order,
            ["left_wrist_roll_joint", "right_wrist_roll_joint"],
            "wrist joint indices",
        )

        self.metrics_path = _resolve_path(cfg_dict.get("metrics_path"))
        if not self.metrics_path:
            self.metrics_path = os.path.abspath(os.path.join(project_root, "logs", "metrics_twist.csv"))

        self.viewer = bool(cfg_dict.get("viewer", self.control_cfg.get("viewer", False)))

        control_dict = (
            OmegaConf.to_container(self.control_cfg, resolve=True)
            if isinstance(self.control_cfg, DictConfig)
            else dict(self.control_cfg or {})
        )
        control_dict = control_dict or {}
        control_dict["viewer"] = self.viewer

        self.simulator = instantiate(self.config.simulator)
        self.sim_action_scale = float(control_dict.get("action_scale", 1.0))
        self.default_angles = np.asarray(self.simulator.default_angles, dtype=np.float32)
        self.active_dof_idx = np.asarray(self.simulator.active_dof_idx, dtype=np.int32)
        self.body_indices_active = _map_active_indices(
            self.active_dof_idx,
            self.body_indices,
            "body joint indices",
        )

        motion_cfg = dict(self.motion_cfg)
        motion_path = _resolve_path(motion_cfg.get("motion_path"))
        if motion_path:
            motion_cfg["motion_path"] = motion_path
        self.motion_cfg = motion_cfg

        ref_dof = int(motion_cfg.get("ref_dof", 23))
        total_dof = int(motion_cfg.get("total_dof", 25))
        gym_idx = bool(motion_cfg.get("gym_idx", True))
        zero_padding_list = list(motion_cfg.get("zero_padding_list", WRIST_IDS_25))

        self.motion_loader = MotionDataset(
            motion_cfg,
            simulator=self.simulator,
            ref_dof=ref_dof,
            total_dof=total_dof,
            gym_idx=gym_idx,
            zero_padding_list=zero_padding_list,
        )
        if self.metrics_path:
            self.motion_loader.set_metrics_file(self.metrics_path)

        policy_path = self.policy_cfg.get("checkpoint") or cfg_dict.get("policy_path")
        self.policy_path = _resolve_path(policy_path) if policy_path else None
        self.policy = None
        if self.policy_path:
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)

        self.action_scale = float(self.policy_cfg.get("action_scale", ACTION_SCALE))
        self.action_clip = self.policy_cfg.get("action_clip", None)
        self.history_len = int(self.policy_cfg.get("history_length", 10))

        self.ankle_idx = [int(x) for x in self.policy_cfg.get("ankle_idx", ANKLE_IDX)]
        self.mimic_obs_total_degrees = int(self.policy_cfg.get("mimic_obs_total_degrees", 33))
        self.mimic_obs_wrist_ids = [int(x) for x in self.policy_cfg.get("mimic_obs_wrist_ids", [27, 32])]
        self.mimic_obs_other_ids = [
            idx for idx in range(self.mimic_obs_total_degrees) if idx not in self.mimic_obs_wrist_ids
        ]

        self.last_action = np.zeros(len(TWIST_G1_23_DOF_NAMES), dtype=np.float32)
        self.default_dof_pos = config.robot.get("default_dof_pos")

        obs_full_dim = len(self.mimic_obs_other_ids) + 3 + 2 + 3 * len(TWIST_G1_23_DOF_NAMES)
        self.obs_full_dim = obs_full_dim
        if self.history_len > 0:
            self.proprio_history = deque(
                [np.zeros(obs_full_dim, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
        else:
            self.proprio_history = deque([], maxlen=0)
        self._pending_wrist_ref = np.zeros(len(self.mimic_obs_wrist_ids), dtype=np.float32)
        self.obs_buf_dict = {}

    def reset(self):
        mujoco.mj_resetData(self.simulator.mujoco_model, self.simulator.mujoco_data)
        self.motion_loader.reset()
        self.last_action.fill(0)
        if self.history_len > 0:
            self.proprio_history = deque(
                [np.zeros(self.obs_full_dim, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
        else:
            self.proprio_history = deque([], maxlen=0)
        obs, wrist_ref = self.compute_observation()
        self._pending_wrist_ref = wrist_ref
        self.obs_buf_dict = {"obs": obs}
        return self.obs_buf_dict
    
    def _reset_envs(self, refresh):
        mujoco.mj_resetData(self.simulator.mujoco_model, self.simulator.mujoco_data)
        self.motion_loader.cur_motion_end = False
        self.last_action.fill(0)
        if self.history_len > 0:
            self.proprio_history = deque(
                [np.zeros(self.obs_full_dim, dtype=np.float32) for _ in range(self.history_len)],
                maxlen=self.history_len,
            )
        else:
            self.proprio_history = deque([], maxlen=0)
        self._pending_wrist_ref = np.zeros(len(self.mimic_obs_wrist_ids), dtype=np.float32)
        # self.simulator.calibrate(refresh=refresh, init_ref_dof_pos=self.default_dof_pos)

    def compute_observation(self):
        self.simulator.get_state()
        sim = self.simulator
        
        # --- 1. Compute Mimic Refernece  ---
        t = self.motion_loader.timestep
        idx = (t + 1) % self.motion_loader.motion.time_step_total
        
        ref_p_w = self.motion_loader.motion.body_pos_w[idx, 0]  # Pelvis
        ref_q_w = self.motion_loader.motion.body_quat_w[idx, 0]
        ref_v_w = self.motion_loader.motion.body_lin_vel_w[idx, 0]
        ref_a_w = self.motion_loader.motion.body_ang_vel_w[idx, 0]
        ref_j = self.motion_loader.motion.joint_pos[idx]
        
        # Align coordinates
        aligned_q_xyzw = self.motion_loader.motion_init_align.align_quat(ref_q_w[[1,2,3,0]])
        aligned_v_w = self.motion_loader.motion_init_align.align_vec_batch(ref_v_w)
        aligned_a_w = self.motion_loader.motion_init_align.align_vec_batch(ref_a_w)
        
        # Switch to the local coordinate system
        # (the strategy requires the perception of the speed relative to the reference frame)
        q_torch = torch.from_numpy(aligned_q_xyzw).to(self.device).float().unsqueeze(0)
        v_torch = torch.from_numpy(aligned_v_w).to(self.device).float().unsqueeze(0)
        a_torch = torch.from_numpy(aligned_a_w).to(self.device).float().unsqueeze(0)
        
        local_v = quat_rotate_inverse(q_torch, v_torch, w_last=True).cpu().numpy().squeeze()
        local_a = quat_rotate_inverse(q_torch, a_torch, w_last=True).cpu().numpy().squeeze()
        ref_rpy = quat_to_euler_wxyz_twist(ref_q_w)

        # Assemble the mimic, obs and disassemble the wrist joint
        mimic_33 = np.concatenate([[ref_p_w[2]], ref_rpy, local_v, [local_a[2]], ref_j]).astype(np.float32)
        if mimic_33.shape[0] != self.mimic_obs_total_degrees:
            raise ValueError(
                f"Unexpected mimic_obs size {mimic_33.shape[0]}, expected {self.mimic_obs_total_degrees}."
            )
        mimic_obs = mimic_33[self.mimic_obs_other_ids]
        wrist_ref = mimic_33[self.mimic_obs_wrist_ids] if self.mimic_obs_wrist_ids else np.zeros(0, dtype=np.float32)

        # --- 2. Compute proprio (74 dims) ---
        cur_rpy = quat_to_euler_wxyz_twist(sim.root_quat[[3, 0, 1, 2]])
        body_pos_23 = sim.dof_pos[self.body_indices_active]
        body_vel_23 = sim.dof_vel[self.body_indices_active]
        obs_v = body_vel_23.copy()
        for idx in self.ankle_idx:
            if 0 <= idx < obs_v.shape[0]:
                obs_v[idx] = 0.0
        
        proprio = np.concatenate([
            sim.base_ang_vel * 0.25, cur_rpy[:2],
            (body_pos_23 - self.default_dof_pos),
            obs_v * 0.05, self.last_action
        ])

        # 3. Concatenate history and update metrics
        obs_full = np.concatenate([mimic_obs, proprio])
        hist_flat = np.array(self.proprio_history, dtype=np.float32).flatten()
        self.proprio_history.append(obs_full)
        self.motion_loader._update_metrics()
        
        return np.concatenate([obs_full, hist_flat]), wrist_ref

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != len(TWIST_G1_23_DOF_NAMES):
            raise ValueError(f"Expected action size {len(TWIST_G1_23_DOF_NAMES)}, got {action.size}.")

        self.last_action = action.copy()
        if self.action_clip is not None:
            action = np.clip(action, -float(self.action_clip), float(self.action_clip))

        target_dof_pos = self.default_angles.copy()
        target_dof_pos[self.body_indices] = action * self.action_scale + self.default_dof_pos
        if self._pending_wrist_ref is not None and self._pending_wrist_ref.size:
            target_dof_pos[self.wrist_indices] = self._pending_wrist_ref

        action_cmd = (target_dof_pos - self.default_angles) / self.sim_action_scale
        if action_cmd.shape[0] != self.simulator.num_action:
            action_cmd = action_cmd[self.active_dof_idx]
        self.simulator.apply_action(action_cmd)

        if getattr(self.simulator, "marker", False):
            markers_world = self._get_reference_markers_world()
            if markers_world is not None:
                self.simulator.update_marker_pos(markers_world[None, ...])

        obs, wrist_ref = self.compute_observation()

        termination_obs = self._check_termination()
        if termination_obs is not None:
            obs, wrist_ref = termination_obs

        self.motion_loader.post_step_callback()
        if self.motion_loader.cur_motion_end:
            self.motion_loader.next_motion(fail=False)
            return self.reset()

        self._pending_wrist_ref = wrist_ref
        self.obs_buf_dict = {"obs": obs}
        return self.obs_buf_dict
    
    def _check_termination(self):
        hard_reset = self.simulator.check_termination()
        if hard_reset:
            self.next_motion(fail=True)
            self._reset_envs(True)
            return self.compute_observation()
        return None

    def next_motion(self, fail: bool = False):
        self.motion_loader.next_motion(fail)
        return self.reset()
    
    def _get_reference_markers_world(self) -> Optional[np.ndarray]:
        """
            Obtain the positions (Nx3) of all joint points of 
            the current reference motion in the world coordinate system
        """
        t = self.motion_loader.timestep
        # motion.body_pos_w.shape: [Frames, Num_Bodies, 3]
        raw_body_pos = self.motion_loader.motion.body_pos_w[t]
        
        markers_world = self.motion_loader.motion_init_align.align_vec_batch(raw_body_pos)
        
        return markers_world.astype(np.float32)
