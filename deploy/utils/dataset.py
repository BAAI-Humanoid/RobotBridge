import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Any

import numpy as np
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
from simulator.base_sim import BaseSim
import csv 

from utils.motion_lib.rotations import quat_error_magnitude, yaw_quat, quat_mul, quat_apply, quat_inverse
from utils.transformation import quat_rotate_inverse
TORSO_INDEX = 15
DESIRED_BODY_INDICES = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]
EEF_INDICES = [3, 6, 10, 13]  # index in DESIRED_BODY_INDICES

class MotionLoader:
    def __init__(self, motion_path: str, body_indexes: Sequence[int], device: str = "cpu", 
                 ref_dof: int = 29, gym_idx: bool = False, zero_padding_list: list = []):
        self._body_indexes = body_indexes
        self.device = device
        self.ref_dof = ref_dof
        self.zero_padding_list = zero_padding_list

        self.file_list = []
        
        # 1. 递归获取所有运动文件
        if os.path.isdir(motion_path):
            for root, dirs, files in os.walk(motion_path):
                for file in files:
                    if file.endswith(".npz"):
                        self.file_list.append(os.path.join(root, file))
            self.file_list.sort()  # 排序以保证顺序一致性
        elif os.path.isfile(motion_path):
            self.file_list = [motion_path]
        else:
            raise FileNotFoundError(f"Invalid path: {motion_path}")
        
        self.current_file_idx = 0
        self.current_file = ""

        self.idx2gym = [
            0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28
        ]
        self.gym_idx = gym_idx
        
        self._load_motion_file(self.file_list[self.current_file_idx])

    def fill_zeros_at_indices(self, data, zero_indices):
        """
        insert zero columns specified by zero_indices
        data: np.array, original data
        zero_indices: list, specify where to fill zeros
        """
        target_dim = data.shape[-1] + len(zero_indices)
        data_indices = [i for i in range(target_dim) if i not in zero_indices]

        if data.ndim == 2:
            res = np.zeros((data.shape[0], target_dim), dtype=data.dtype)
            res[:, data_indices] = data
        else:
            res = np.zeros(target_dim, dtype=data.dtype)
            res[data_indices] = data
        return res

    def _load_motion_file(self, motion_file: str):
        """内部函数：负责读取具体的 npz 文件数据"""
        logger.info(f"Loading motion file [{self.current_file_idx + 1}/{len(self.file_list)}]: {motion_file}")
        self.current_file = motion_file
        data = np.load(motion_file)
        
        self.fps = data["fps"]
        if self.gym_idx:
            self.joint_pos = data["joint_pos"][1:, self.idx2gym]
            self.joint_vel = data["joint_vel"][1:, self.idx2gym]
        else:
            self.joint_pos = data["joint_pos"][1:]
            self.joint_vel = data["joint_vel"][1:]

        if self.ref_dof == 23:
            self.joint_pos = np.concatenate((self.joint_pos[:, :19], self.joint_pos[:, 22:26]), axis=-1)
            self.joint_vel = np.concatenate((self.joint_vel[:, :19], self.joint_vel[:, 22:26]), axis=-1)

        if len(self.zero_padding_list) > 0:
            self.joint_pos = self.fill_zeros_at_indices(self.joint_pos, self.zero_padding_list)
            self.joint_vel = self.fill_zeros_at_indices(self.joint_vel, self.zero_padding_list)

        self._body_pos_w = data["body_pos_w"][1:]
        self._body_quat_w = data["body_quat_w"][1:]
        self._body_lin_vel_w = data["body_lin_vel_w"][1:]
        self._body_ang_vel_w = data["body_ang_vel_w"][1:]
        self.time_step_total = self.joint_pos.shape[0]

    def next_motion(self) -> bool:
        """切换到下一个运动文件。如果结束则退出程序。"""
        self.current_file_idx += 1
        if self.current_file_idx >= len(self.file_list):
            print("All motion files processed")
            import sys
            sys.exit(0) # 退出程序
        
        self._load_motion_file(self.file_list[self.current_file_idx])
        return True

    @property
    def body_pos_w(self) -> np.ndarray:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> np.ndarray:
        "xyzw"
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> np.ndarray:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> np.ndarray:
        return self._body_ang_vel_w[:, self._body_indexes]
    


class MosaicMetaParsingError(RuntimeError):
    """Raised when ONNX metadata required for Mosaic is incomplete."""


def _parse_str_list(value: Optional[str]) -> List[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_float_array(value: Optional[str]) -> np.ndarray:
    if value is None or value.strip() == "":
        return np.zeros(0, dtype=np.float32)
    return np.asarray([float(item) for item in value.split(",") if item.strip()], dtype=np.float32)

@dataclass()
class MosaicModelMeta:
    joint_names: List[str]
    default_joint_pos: np.ndarray
    joint_stiffness: np.ndarray
    joint_damping: np.ndarray
    action_scale: np.ndarray
    body_names: List[str]
    anchor_body_name: str

    @classmethod
    def from_onnx_session(cls, session) -> "MosaicModelMeta":
        """Parse Mosaic-specific metadata from an ONNX session."""
        meta = session.get_modelmeta()
        meta_map: Dict[str, str] = meta.custom_metadata_map or {}

        joint_names = _parse_str_list(meta_map.get("joint_names"))
        default_joint_pos = _parse_float_array(meta_map.get("default_joint_pos"))
        joint_stiffness = _parse_float_array(meta_map.get("joint_stiffness"))
        joint_damping = _parse_float_array(meta_map.get("joint_damping"))
        action_scale = _parse_float_array(meta_map.get("action_scale"))
        body_names = _parse_str_list(meta_map.get("body_names"))
        anchor_body_name = meta_map.get("anchor_body_name")

        required_fields = {
            "joint_names": joint_names,
            "default_joint_pos": default_joint_pos,
            "joint_stiffness": joint_stiffness,
            "joint_damping": joint_damping,
            "action_scale": action_scale,
            "body_names": body_names,
            "anchor_body_name": anchor_body_name,
        }

        for key, value in required_fields.items():
            if value is None or (isinstance(value, (list, tuple)) and len(value) == 0) or (
                isinstance(value, np.ndarray) and value.size == 0
            ):
                raise MosaicMetaParsingError(f"ONNX metadata is missing `{key}` required by Mosaic.")

        def _ensure_length(arr: np.ndarray, name: str) -> np.ndarray:
            if arr.size != len(joint_names):
                raise MosaicMetaParsingError(
                    f"Metadata field `{name}` expects {len(joint_names)} values, got {arr.size}."
                )
            return arr.astype(np.float32)

        default_joint_pos = _ensure_length(default_joint_pos, "default_joint_pos")
        joint_stiffness = _ensure_length(joint_stiffness, "joint_stiffness")
        joint_damping = _ensure_length(joint_damping, "joint_damping")
        action_scale = _ensure_length(action_scale, "action_scale")

        if anchor_body_name not in body_names:
            raise MosaicMetaParsingError(
                f"Anchor body `{anchor_body_name}` is not present in body names metadata."
            )

        return cls(
            joint_names=joint_names,
            default_joint_pos=default_joint_pos,
            joint_stiffness=joint_stiffness,
            joint_damping=joint_damping,
            action_scale=action_scale,
            body_names=body_names,
            anchor_body_name=anchor_body_name,
        )

    def joint_index_map(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.joint_names)}

    def to_joint_order(self, ordered_joint_names: Sequence[str]) -> Dict[str, np.ndarray]:
        """Reorder metadata arrays to match a target joint order."""
        index_map = self.joint_index_map()
        indices = []
        for joint in ordered_joint_names:
            if joint not in index_map:
                raise MosaicMetaParsingError(
                    f"Joint `{joint}` is missing in Mosaic metadata. "
                    "Ensure the policy was exported with complete metadata."
                )
            indices.append(index_map[joint])
        indices_arr = np.asarray(indices, dtype=np.int64)
        return {
            "default_joint_pos": self.default_joint_pos[indices_arr],
            "joint_stiffness": self.joint_stiffness[indices_arr],
            "joint_damping": self.joint_damping[indices_arr],
            "action_scale": self.action_scale[indices_arr],
        }


class TransformAligner:
    """Yaw-only SE(3) alignment helper used by Mosaic motions."""

    def __init__(self, yaw_only: bool = True, xy_only: bool = True) -> None:
        self.yaw_only = yaw_only
        self.xy_only = xy_only
        self._base_rot = sRot.identity()
        self._base_pos = np.zeros(3, dtype=np.float64)

    def set_base(self, quat: np.ndarray, pos: np.ndarray) -> None:
        rot = sRot.from_quat(quat)
        if self.yaw_only:
            euler = rot.as_euler("xyz")
            euler[0] = 0.0
            euler[1] = 0.0
            rot = sRot.from_euler("xyz", euler)
        self._base_rot = rot

        base_pos = np.asarray(pos, dtype=np.float64)
        if self.xy_only:
            base_pos = base_pos.copy()
            base_pos[2] = 0.0
        self._base_pos = base_pos
        logger.debug(f"TransformAligner base set to pos={self._base_pos}, yaw_only={self.yaw_only}")

    def align_quat(self, quat: np.ndarray) -> np.ndarray:
        current = sRot.from_quat(quat)
        aligned = self._base_rot.inv() * current
        return aligned.as_quat()
    
    def align_quat_batch(self, quat: np.ndarray) -> np.ndarray:
        """Batch version of align_quat.

        Args:
            quat: (N, 4) or (..., 4) quaternions in the original motion/world frame.
        Returns:
            aligned quaternions with the same leading shape as input.
        """
        quat = np.asarray(quat, dtype=np.float64)
        if quat.shape[-1] != 4:
            raise ValueError(f"Expected last dim == 4 for quaternions, got shape={quat.shape}")
        flat = quat.reshape(-1, 4)
        current = sRot.from_quat(flat)
        aligned = self._base_rot.inv() * current
        return aligned.as_quat().reshape(quat.shape).astype(np.float32)

    def align_pos(self, pos: np.ndarray) -> np.ndarray:
        pos = np.asarray(pos, dtype=np.float64)
        rel = pos - self._base_pos
        return self._base_rot.inv().apply(rel)

    def align_pos_batch(self, pos: np.ndarray) -> np.ndarray:
        """Batch version of align_pos.

        Args:
            pos: (N, 3) or (..., 3) positions in the original motion/world frame.
        Returns:
            aligned positions with the same leading shape as input.
        """
        pos = np.asarray(pos, dtype=np.float64)
        if pos.shape[-1] != 3:
            raise ValueError(f"Expected last dim == 3 for positions, got shape={pos.shape}")
        flat = pos.reshape(-1, 3)
        rel = flat - self._base_pos.reshape(1, 3)
        aligned = self._base_rot.inv().apply(rel)
        return aligned.reshape(pos.shape).astype(np.float32)

    def align_vec_batch(self, vec: np.ndarray) -> np.ndarray:
        """Batch version of alignment for pure vectors (no translation)."""
        vec = np.asarray(vec, dtype=np.float64)
        if vec.shape[-1] != 3:
            raise ValueError(f"Expected last dim == 3 for vectors, got shape={vec.shape}")
        flat = vec.reshape(-1, 3)
        aligned = self._base_rot.inv().apply(flat)
        return aligned.reshape(vec.shape).astype(np.float32)

    def align_transform(self, quat: np.ndarray, pos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.align_quat(quat), self.align_pos(pos)


class MotionDataset:
    """Light-weight loader for Mosaic motion `.npz` files."""

    def __init__(self, motion_cfg: Dict[str, Any], simulator: BaseSim, ref_dof: int = 29, total_dof: int = 29, gym_idx: bool = False, zero_padding_list: list = []):

        logger.info(f"Loading Mosaic motion: {motion_cfg['motion_path']}")
        self.simulator = simulator

        self.override_robot_anchor_pos = motion_cfg['override_robot_anchor_pos']
        self.interp_steps = int(motion_cfg.get("interp_steps", 0))
        body_indexes = [motion_cfg['body_names_all'].index(name) for name in motion_cfg['body_names']]
        self.motion_anchor_body_index = motion_cfg['body_names'].index(motion_cfg['anchor_body_name'])
        self.command_horizon = motion_cfg.get("command_horizon", 1)
        self.command_velocity = motion_cfg.get("command_velocity", True)

        self.motion = MotionLoader(motion_cfg['motion_path'], body_indexes, ref_dof=ref_dof, gym_idx=gym_idx, zero_padding_list=zero_padding_list)
        self.ref_dof = ref_dof
        self.total_dof = total_dof

        self.timestep = 0
        self.playing = False
        self.motion_init_align = TransformAligner(yaw_only=True, xy_only=True)
        self.reset()

        self.metrics = {}
        self.metrics_file = None
        self._header_written = False
        self._metrics_no_average = {"feet_stumble_count"}

        self.motion_cfg = motion_cfg
        self.body_indexes = [motion_cfg['body_names_all'].index(name) for name in motion_cfg['body_names']]

        self.cur_motion_end = False

    def set_metrics_file(self, file_path: str):
        self.metrics_file = file_path
        # If the file exists, throw error
        # if os.path.exists(self.metrics_file):
        #     raise RuntimeError("[MotionDataset::set_metrics_file] The eval file already exists.")
        
        parent_dir = os.path.dirname(self.metrics_file)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
            logger.info(f"Created directory for metrics: {parent_dir}")

        with open(self.metrics_file, 'w', encoding='utf-8') as f:
            pass 

        self._header_written = False
        logger.info(f"Metrics file initialized: {self.metrics_file}")

    @property
    def command(self) -> np.ndarray:
        # clip the valid range of the command
        valid_range = np.clip(range(self.timestep, self.timestep + self.command_horizon), 0, self.motion.time_step_total - 1)
        joint_pos_seq = self.motion.joint_pos[valid_range]
        if self.command_velocity:
            joint_vel_seq = self.motion.joint_vel[valid_range]
            command_seq = np.concatenate([joint_pos_seq, joint_vel_seq], axis=-1)
        else:
            command_seq = joint_pos_seq
        return command_seq.reshape(-1)

    @property
    def joint_pos(self) -> np.ndarray:
        return self.motion.joint_pos[self.timestep].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.motion.joint_vel[self.timestep].copy()

    @property
    def anchor_pos_w(self) -> np.ndarray:
        anchor_pos_w_raw = self.motion.body_pos_w[self.timestep, self.motion_anchor_body_index].copy()
        anchor_pos_w = self.motion_init_align.align_pos(anchor_pos_w_raw)
        return anchor_pos_w

    @property
    def anchor_quat_w(self) -> np.ndarray:
        anchor_quat_w_raw = self.motion.body_quat_w[self.timestep, self.motion_anchor_body_index].copy()[[1, 2, 3, 0]]
        return self.motion_init_align.align_quat(anchor_quat_w_raw)
    
    @property
    def anchor_lin_vel_w(self) -> np.ndarray:
        return self.motion.body_lin_vel_w[self.timestep, self.motion_anchor_body_index].copy()

    @property
    def anchor_ang_vel_w(self) -> np.ndarray:
        return self.motion.body_ang_vel_w[self.timestep, self.motion_anchor_body_index].copy()
    
    @property
    def body_pos_w(self) -> np.ndarray:
        return self.motion.body_pos_w[self.timestep]

    @property
    def body_quat_w(self) -> np.ndarray:
        return self.motion.body_quat_w[self.timestep]

    @property
    def body_lin_vel_w(self) -> np.ndarray:
        return self.motion.body_lin_vel_w[self.timestep]

    @property
    def body_ang_vel_w(self) -> np.ndarray:
        return self.motion.body_ang_vel_w[self.timestep]

    @property
    def robot_anchor_pos_w(self) -> np.ndarray:
        if self.override_robot_anchor_pos:  # OVERRIDE
            return self.anchor_pos_w
        else:
            base_pos = self.simulator.torso_trans
            assert base_pos is not None
            return base_pos

    @property
    def robot_anchor_quat_w(self) -> np.ndarray:
        torso_quat = self.simulator.torso_quat
        assert torso_quat is not None
        return torso_quat
    
    @property
    def robot_body_pos_w(self) -> np.ndarray:
        return self.simulator.robot_fk_info[DESIRED_BODY_INDICES, :3]

    @property
    def robot_body_quat_w(self) -> np.ndarray:
        return self.simulator.robot_fk_info[DESIRED_BODY_INDICES, 3:7]

    @property
    def robot_body_lin_vel_w(self) -> np.ndarray:
        return self.simulator.robot_fk_info[DESIRED_BODY_INDICES, 7:10]

    @property
    def robot_body_ang_vel_w(self) -> np.ndarray:
        return self.simulator.robot_fk_info[DESIRED_BODY_INDICES, 10:]

    @property
    def robot_anchor_lin_vel_w(self) -> np.ndarray:
        return self.simulator.robot_fk_info[TORSO_INDEX, 7:10]

    @property
    def robot_anchor_ang_vel_w(self) -> np.ndarray:
        return self.simulator.robot_fk_info[TORSO_INDEX, 10:]
    
    @property
    def robot_joint_pos(self) -> np.ndarray:
        return self.simulator.dof_pos
    
    @property
    def robot_joint_vel(self) -> np.ndarray:
        return self.simulator.dof_vel
    
    def reset(self):
        self.timestep = 0
        init2anchor_pos = self.motion.body_pos_w[0, self.motion_anchor_body_index].copy()
        init2anchor_quat = self.motion.body_quat_w[0, self.motion_anchor_body_index].copy()[[1, 2, 3, 0]]
        # keep yaw only
        self.motion_init_align.set_base(quat=init2anchor_quat, pos=init2anchor_pos)

        self._metrics_accumulator = {} 

    @property
    def body_pos_w_aligned(self) -> np.ndarray:
        """Aligned reference body positions at current timestep.

        Shape: (num_bodies_selected, 3)
        """
        body_pos_raw = self.motion.body_pos_w[self.timestep].copy()
        return self.motion_init_align.align_pos_batch(body_pos_raw)
    
    @property
    def body_quat_w_aligned(self) -> np.ndarray:
        """Aligned reference body positions at current timestep.

        Shape: (num_bodies_selected, 4)
        """
        body_quat_raw = self.motion.body_quat_w[self.timestep].copy()[..., [1, 2, 3, 0]]
        return self.motion_init_align.align_quat_batch(body_quat_raw)

    def post_step_callback(self):
        self.timestep += 1
        self.cur_motion_end = (self.timestep == self.motion.time_step_total)
        self.timestep = np.clip(self.timestep, 0, self.motion.time_step_total - 1)

    def get_data(self):
        ctrl_data = {
            "command": self.command,
            "joint_pos": self.joint_pos,
            "robot_anchor_pos_w": self.robot_anchor_pos_w,
            "robot_anchor_quat_w": self.robot_anchor_quat_w,
            "anchor_pos_w": self.anchor_pos_w,
            "anchor_quat_w": self.anchor_quat_w,
            "body_pos_w_aligned": self.body_pos_w_aligned,
            "timestep": self.timestep,
        }

        return ctrl_data


    def _compute_feet_stumble(self) -> float:
        forces_w = getattr(self.simulator, "foot_contact_forces_w", None)
        if forces_w is None:
            return 0.0
        forces_w = np.asarray(forces_w)
        if forces_w.size == 0:
            return 0.0
        forces_z = np.abs(forces_w[:, 2])
        forces_xy = np.linalg.norm(forces_w[:, :2], axis=1)
        return float(np.any(forces_xy > 4.0 * forces_z))
    
    def _update_metrics_single_frame(self):
        # import pdb; pdb.set_trace()
        robot_anchor_pos_w_tensor = torch.from_numpy(self.robot_anchor_pos_w)
        robot_anchor_quat_w_tensor = torch.from_numpy(self.robot_anchor_quat_w)
        anchor_pos_w_tensor = torch.from_numpy(self.anchor_pos_w)
        anchor_quat_w_tensor = torch.from_numpy(self.anchor_quat_w)
        body_pos_w_tensor = torch.from_numpy(self.body_pos_w_aligned)
        body_quat_w_tensor = torch.from_numpy(self.body_quat_w_aligned)

        delta_pos_w = robot_anchor_pos_w_tensor
        delta_pos_w[..., 2] = anchor_pos_w_tensor[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_tensor, quat_inverse(anchor_quat_w_tensor, w_last=True), w_last=True)).unsqueeze(0).expand(body_quat_w_tensor.shape[0], 4)

        self.body_quat_relative_w = quat_mul(delta_ori_w, body_quat_w_tensor, w_last=True)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, body_pos_w_tensor - anchor_pos_w_tensor, w_last=True)

        anchor_lin_vel_b = quat_rotate_inverse(np.roll(self.anchor_quat_w, 1), self.anchor_lin_vel_w)
        robot_anchor_lin_vel_b = quat_rotate_inverse(np.roll(self.robot_anchor_quat_w, 1), self.robot_anchor_lin_vel_w)

        self.metrics["error_anchor_pos"] = np.linalg.norm(self.anchor_pos_w - self.robot_anchor_pos_w, axis=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(torch.from_numpy(self.anchor_quat_w[[3, 0, 1, 2]]), torch.from_numpy(self.robot_anchor_quat_w[[3, 0, 1, 2]]))
        self.metrics["error_anchor_lin_vel"] = np.linalg.norm(anchor_lin_vel_b - robot_anchor_lin_vel_b)
        self.metrics["error_anchor_ang_vel"] = np.linalg.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, axis=-1)
        self.metrics["error_body_pos"] = np.linalg.norm(self.body_pos_relative_w - self.robot_body_pos_w, axis=-1).mean(axis=-1)
        self.metrics["error_eef_pos"] = np.linalg.norm(self.body_pos_relative_w[EEF_INDICES] - self.robot_body_pos_w[EEF_INDICES], axis=-1).mean(axis=-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w[:, [3, 0, 1, 2]], torch.from_numpy(self.robot_body_quat_w[:, [3, 0, 1, 2]])).mean(axis=-1)
        
        self.metrics["error_body_pos_w"] = np.linalg.norm(self.body_pos_w_aligned - self.robot_body_pos_w, axis=-1).mean(axis=-1)
        self.metrics["error_eef_pos_w"] = np.linalg.norm(self.body_pos_w_aligned[EEF_INDICES] - self.robot_body_pos_w[EEF_INDICES], axis=-1).mean(axis=-1)
        self.metrics["error_body_rot_w"] = quat_error_magnitude(torch.from_numpy(self.body_quat_w_aligned[:, [3, 0, 1, 2]]), torch.from_numpy(self.robot_body_quat_w[:, [3, 0, 1, 2]])).mean(axis=-1)

        self.metrics["error_body_lin_vel"] = np.linalg.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, axis=-1).mean(axis=-1)
        self.metrics["error_body_ang_vel"] = np.linalg.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, axis=-1).mean(axis=-1)

        if not self.motion.gym_idx:
            joint_pos = self.joint_pos[..., self.motion.idx2gym]
            joint_vel = self.joint_vel[..., self.motion.idx2gym]
        else:
            joint_pos = self.joint_pos
            joint_vel = self.joint_vel

        self.metrics["error_joint_pos"] = np.abs(joint_pos - self.robot_joint_pos).mean(axis=-1)
        self.metrics["error_joint_vel"] = np.abs(joint_vel - self.robot_joint_vel).mean(axis=-1)
        if "feet_stumble_count" not in self.metrics:
            self.metrics["feet_stumble_count"] = 0.0
        if "feet_stumble_ratio" not in self.metrics:
            num_steps = self.timestep + 1
            self.metrics["feet_stumble_ratio"] = self.metrics["feet_stumble_count"] / max(num_steps, 1)

    def _update_metrics(self):
        robot_anchor_pos_w_tensor = torch.from_numpy(self.robot_anchor_pos_w)
        robot_anchor_quat_w_tensor = torch.from_numpy(self.robot_anchor_quat_w)
        anchor_pos_w_tensor = torch.from_numpy(self.anchor_pos_w)
        anchor_quat_w_tensor = torch.from_numpy(self.anchor_quat_w)
        body_pos_w_tensor = torch.from_numpy(self.body_pos_w_aligned)
        body_quat_w_tensor = torch.from_numpy(self.body_quat_w_aligned)

        delta_pos_w = robot_anchor_pos_w_tensor.clone()
        delta_pos_w[..., 2] = anchor_pos_w_tensor[..., 2]
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_tensor, quat_inverse(anchor_quat_w_tensor, w_last=True), w_last=True)).unsqueeze(0).expand(body_quat_w_tensor.shape[0], 4)

        body_quat_relative_w = quat_mul(delta_ori_w, body_quat_w_tensor, w_last=True)
        body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, body_pos_w_tensor - anchor_pos_w_tensor, w_last=True)

        anchor_lin_vel_b = quat_rotate_inverse(np.roll(self.anchor_quat_w, 1), self.anchor_lin_vel_w)
        robot_anchor_lin_vel_b = quat_rotate_inverse(np.roll(self.robot_anchor_quat_w, 1), self.robot_anchor_lin_vel_w)

        curr_err = {}
        # Anchor 
        curr_err["error_anchor_pos"] = np.linalg.norm(self.anchor_pos_w - self.robot_anchor_pos_w)
        curr_err["error_anchor_rot"] = quat_error_magnitude(
            torch.from_numpy(self.anchor_quat_w[[3, 0, 1, 2]]), 
            torch.from_numpy(self.robot_anchor_quat_w[[3, 0, 1, 2]])
        ).item()
        curr_err["error_anchor_lin_vel"] = np.linalg.norm(anchor_lin_vel_b - robot_anchor_lin_vel_b)
        curr_err["error_anchor_ang_vel"] = np.linalg.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w)
        
        # Body
        curr_err["error_body_pos"] = np.linalg.norm(body_pos_relative_w.numpy() - self.robot_body_pos_w, axis=-1).mean(axis=-1)
        curr_err["error_eef_pos"] = np.linalg.norm(body_pos_relative_w[EEF_INDICES] - self.robot_body_pos_w[EEF_INDICES], axis=-1).mean(axis=-1)
        curr_err["error_body_rot"] = quat_error_magnitude(
            body_quat_relative_w[:, [3, 0, 1, 2]], 
            torch.from_numpy(self.robot_body_quat_w[:, [3, 0, 1, 2]])
        ).mean().item()

        curr_err["error_body_pos_w"] = np.linalg.norm(self.body_pos_w_aligned - self.robot_body_pos_w, axis=-1).mean(axis=-1)
        curr_err["error_eef_pos_w"] = np.linalg.norm(self.body_pos_w_aligned[EEF_INDICES] - self.robot_body_pos_w[EEF_INDICES], axis=-1).mean(axis=-1)
        curr_err["error_body_rot_w"] = quat_error_magnitude(body_quat_w_tensor[:, [3, 0, 1, 2]], torch.from_numpy(self.robot_body_quat_w[:, [3, 0, 1, 2]])).mean(axis=-1)

        curr_err["error_body_lin_vel"] = np.linalg.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, axis=-1).mean(axis=-1)
        curr_err["error_body_ang_vel"] = np.linalg.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, axis=-1).mean(axis=-1)

        # Joint
        if not self.motion.gym_idx:
            joint_pos = self.joint_pos[..., self.motion.idx2gym]
            joint_vel = self.joint_vel[..., self.motion.idx2gym]
        else:
            joint_pos = self.joint_pos
            joint_vel = self.joint_vel

        curr_err["error_joint_pos"] = np.abs(joint_pos - self.robot_joint_pos).mean(axis=-1)
        curr_err["error_joint_vel"] = np.abs(joint_vel - self.robot_joint_vel).mean(axis=-1)
        curr_err["feet_stumble_count"] = self._compute_feet_stumble()

        num_steps = self.timestep + 1
        
        for key, val in curr_err.items():
            if key not in self._metrics_accumulator:
                self._metrics_accumulator[key] = 0.0
            self._metrics_accumulator[key] += val

            if key in self._metrics_no_average:
                self.metrics[key] = self._metrics_accumulator[key]
            else:
                self.metrics[key] = self._metrics_accumulator[key] / num_steps

        if "feet_stumble_count" in self._metrics_accumulator:
            self.metrics["feet_stumble_ratio"] = self._metrics_accumulator["feet_stumble_count"] / max(num_steps, 1)

    def _write_metrics_to_csv(self, fail: bool = False):
        if fail:
            self._update_metrics_single_frame()
        # Convert tensor or numpy scalar to Python float
        row_data = {}
        for k, v in self.metrics.items():
            if isinstance(v, (torch.Tensor, np.ndarray)):
                row_data[k] = v.item() if v.size == 1 else v.mean().item()
            else:
                row_data[k] = v
        
        # Add timestamp
        row_data["step"] = self.timestep
        row_data["success"] = float(not fail)
        
        with open(self.metrics_file, mode='a', newline='', encoding='utf-8') as f:
            # define fieldnames when first write
            fieldnames = sorted(row_data.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not self._header_written:
                writer.writeheader()
                self._header_written = True
                
            writer.writerow(row_data)

    def next_motion(self, fail: bool = False):
        """调用此函数切换到下一个动作文件"""
        # 1. 让 loader 切换文件
        self._write_metrics_to_csv(fail)
        self.motion.next_motion()
        
        logger.success(f"Successfully switched to: {os.path.basename(self.motion.current_file)}")

"""
0: pelvis
1: left_hip_pitch_link
2: left_hip_roll_link
3: left_hip_yaw_link
4: left_knee_link
5: left_ankle_pitch_link
6: left_ankle_roll_link
7: right_hip_pitch_link
8: right_hip_roll_link
9: right_hip_yaw_link
10: right_knee_link
11: right_ankle_pitch_link
12: right_ankle_roll_link
13: waist_yaw_link
14: waist_roll_link
15: torso_link
16: left_shoulder_pitch_link
17: left_shoulder_roll_link
18: left_shoulder_yaw_link
19: left_elbow_link
20: left_wrist_roll_link
21: left_wrist_pitch_link
22: left_wrist_yaw_link
23: right_shoulder_pitch_link
24: right_shoulder_roll_link
25: right_shoulder_yaw_link
26: right_elbow_link
27: right_wrist_roll_link
28: right_wrist_pitch_link
29: right_wrist_yaw_link
"""