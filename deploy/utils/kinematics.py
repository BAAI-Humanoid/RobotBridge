import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np
from loguru import logger


@dataclass
class ForwardKinematicsConfig:
    """Forward Kinematics Config"""

    xml_path: str
    debug_viz: bool = False
    kinematic_joint_names: Optional[List[str]] = None

    @classmethod
    def from_asset_cfg(
        cls, asset_root: str, asset_file: str, joint_names: Optional[List[str]] = None, debug_viz: bool = False
    ) -> "ForwardKinematicsConfig":
        """asset_root and asset_file are used to load the model"""
        xml_path = os.path.join(asset_root, asset_file)
        return cls(xml_path=xml_path, debug_viz=debug_viz, kinematic_joint_names=joint_names)


class MujocoKinematics:
    """Forward kinematics tool for MuJoCo"""

    def __init__(self, cfg: ForwardKinematicsConfig):
        self.cfg = cfg
        self.model = mujoco.MjModel.from_xml_path(self.cfg.xml_path)  # pyright: ignore[reportAttributeAccessIssue]
        self.data = mujoco.MjData(self.model)  # pyright: ignore[reportAttributeAccessIssue]
        logger.debug(f"[MujocoKinematics] loaded model from {self.cfg.xml_path}")

        self.has_free_joint = (
            self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE if self.model.njnt > 0 else False  # pyright: ignore[reportAttributeAccessIssue]
        )
        self.qpos_offset = 7 if self.has_free_joint else 0

        self.num_bodies = self.model.nbody
        self.body_offset = 1 if self.has_free_joint else 0
        self.body_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)  # pyright: ignore[reportAttributeAccessIssue]
            for i in range(self.body_offset, self.num_bodies)
        ]

        self.debug_viz = self.cfg.debug_viz
        self.viewer = None
        if self.debug_viz:
            try:
                import mujoco_viewer

                self.viewer = mujoco_viewer.MujocoViewer(
                    self.model,
                    self.data,
                    width=900,
                    height=900,
                    hide_menus=True,
                )
                self.viewer.cam.distance = 3.0
                self.viewer._render_every_frame = True
                self._debug_viz_last_render_time = 0.0
            except Exception as exc:  # pragma: no cover - only for debugging
                logger.warning(f"[MujocoKinematics] debug_viz initialization failed, visualization will be disabled: {exc}")
                self.debug_viz = False
                self.viewer = None

        self.update_joint_names_subset(self.cfg.kinematic_joint_names)

    def update_joint_names_subset(self, joint_names_subset: Optional[List[str]] = None) -> None:
        """set the subset of joints to use, if not provided, use all joints (except the root free joint)"""
        if joint_names_subset is not None:
            self.joint_names = list(joint_names_subset)
        else:
            joint_names: List[str] = []
            for i in range(self.model.njnt):
                if self.has_free_joint and i == 0:
                    continue
                joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)  # pyright: ignore[reportAttributeAccessIssue]
                joint_names.append(joint_name)
            self.joint_names = joint_names

        self.num_joints = len(self.joint_names)

        self.joint_qpos_indices: List[int] = []
        for joint_name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)  # pyright: ignore[reportAttributeAccessIssue]
            if joint_id == -1:
                raise ValueError(f"Joint {joint_name} not found in the model.")
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.joint_qpos_indices.append(int(qpos_addr))

        logger.debug(f"[MujocoKinematics] set fk with {self.num_joints} joints")

    def forward(
        self,
        joint_pos: np.ndarray,
        base_pos: Optional[np.ndarray] = None,
        base_quat: Optional[np.ndarray] = None,
        joint_vel: Optional[np.ndarray] = None,
        base_lin_vel: Optional[np.ndarray] = None,
        base_ang_vel: Optional[np.ndarray] = None,
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """
        perform one forward kinematics, return the pose and velocity information of each body
        parameters: base_quat is input as XYZW, and the function will convert it to WXYZ internally
        """
        assert joint_pos.shape[0] == self.num_joints, (
            f"Expected joint_pos of shape ({self.num_joints},), got {joint_pos.shape}"
        )

        qpos_full = np.zeros(self.model.nq, dtype=np.float64)
        if self.has_free_joint:
            if base_pos is not None:
                if base_pos.shape != (3,):
                    raise ValueError("base_pos must be of shape (3,)")
                qpos_full[0:3] = base_pos
            if base_quat is not None:
                if base_quat.shape != (4,):
                    raise ValueError("base_quat must be of shape (4,)")
                qpos_full[3:7] = base_quat[[3, 0, 1, 2]]  # xyzw -> wxyz
        qpos_full[self.joint_qpos_indices] = joint_pos
        self.data.qpos[:] = qpos_full

        if joint_vel is not None or base_lin_vel is not None or base_ang_vel is not None:
            qvel_full = np.zeros(self.model.nv, dtype=np.float64)
            if self.has_free_joint:
                if base_lin_vel is not None:
                    qvel_full[0:3] = base_lin_vel
                if base_ang_vel is not None:
                    qvel_full[3:6] = base_ang_vel
                offset = 6
            else:
                offset = 0
            if joint_vel is not None:
                qvel_full[offset : offset + self.num_joints] = joint_vel
            self.data.qvel[:] = qvel_full

        mujoco.mj_forward(self.model, self.data)  # pyright: ignore[reportAttributeAccessIssue]

        offset = 1 if self.has_free_joint else 0
        body_info: Dict[str, Dict[str, np.ndarray]] = {}
        body_info_tensor = None
        for i in range(offset, self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)  # pyright: ignore[reportAttributeAccessIssue]
            pos = self.data.xpos[i].copy()
            quat = self.data.xquat[i].copy()[[1, 2, 3, 0]]  # wxyz -> xyzw
            lin_vel = self.data.cvel[i].copy()[3:]
            ang_vel = self.data.cvel[i].copy()[0:3]
            body_info[name] = {
                "pos": pos,
                "quat": quat,
                "lin_vel": lin_vel,
                "ang_vel": ang_vel,
            }

            data = np.concatenate([pos, quat, lin_vel, ang_vel], axis=-1)
            data = np.expand_dims(data, axis=0)
            body_info_tensor = np.concatenate([body_info_tensor, data], axis=0) if body_info_tensor is not None else data
        # if self.debug_viz and self.viewer is not None and getattr(self.viewer, "is_alive", True):
        #     # simple visualization rendering
        #     self.viewer.cam.lookat = self.data.qpos.astype(np.float32)[:3]
        #     self.viewer.render()

        return body_info, body_info_tensor

    def __del__(self):
        try:
            if self.debug_viz and self.viewer is not None:
                self.viewer.close()
        except Exception:
            pass
