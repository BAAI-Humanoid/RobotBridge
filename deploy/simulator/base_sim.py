


import numpy as np
import time
import select
import threading
from loguru import logger

from scipy.spatial.transform import Rotation as sRot

import sys
sys.path.append('../')

from utils.kinematics import MujocoKinematics, ForwardKinematicsConfig
from unitree_sdk2.lcm_types.camera_reference_data_lcmt import camera_reference_data_lcmt

class BaseSim:
    def __init__(self, config):
        self.cfg = config
        self.torso_name = self.cfg.asset.torso_name
        self.kinematic_cfg = ForwardKinematicsConfig.from_asset_cfg(
            asset_root=self.cfg.asset.asset_root,
            asset_file=self.cfg.asset.asset_file,
            joint_names=self.cfg.asset.kinematic_joint_names,
        )

        self.kinematic = MujocoKinematics(self.kinematic_cfg)
        self._setup()
        self._load_asset()
        self._init_low_state()

        self._init_teleop = False
        self.sync = True

    def _setup(self):
        self.low_dt = self.cfg.control.low_dt
        self.decimation = self.cfg.control.decimation
        self.high_dt = self.low_dt * self.decimation

        logger.info(f'Robot-level Control Frequency Set to {1/self.low_dt}HZ')
        logger.info(f'Policy-level Control Frequency Set to {1/self.high_dt}HZ')

    def _load_asset(self):
        self.default_angles = np.array(self.cfg.asset.default_angles, dtype=np.float32)
        self.kps = np.array(self.cfg.asset.kps, dtype=np.float32)
        self.kds = np.array(self.cfg.asset.kds, dtype=np.float32)
        self.dof_names = list(self.cfg.asset.joint_order.keys())
        self.frozen_dof_names = self.cfg.asset.frozen_dof_names
        self.active_dof_idx = np.array([i for i in range(len(self.dof_names)) if self.dof_names[i] not in self.frozen_dof_names])
        self.num_dof = self.cfg.asset.num_dof
        self.num_action = self.cfg.asset.num_action
        
        for i,k in enumerate(self.dof_names):
            logger.info(f'Joint {k}  Default Angles {self.default_angles[i]}  P_gain {self.kps[i]}  D_gain {self.kds[i]} ')
        logger.info(f'Total Number of dof: {self.num_dof}')
        logger.info(f'Number of Action: {self.num_action}')

        self.joint_serial_num = self.cfg.asset.joint_order
        # index for policy to robot
        self.idx_r2p=[self.joint_serial_num[k] for k in self.joint_serial_num]
        # index for robot to policy
        self.idx_p2r=[self.idx_r2p.index(i) for i in range(self.num_dof)]

    def _init_low_state(self):
        self.dof_tracking_init_pos = np.zeros((1, self.num_action), dtype=np.float32)
        self.root_trans = np.zeros((1,3), dtype=np.float32)
        self.root_quat = np.zeros((1,4), dtype=np.float32)
        self.torso_quat = np.zeros((1,4), dtype=np.float32)
        self.torso_trans = np.zeros((1,3), dtype=np.float32)
        self.root_trans_world = np.zeros((1,3), dtype=np.float32)
        self.root_trans_world_tmp = np.zeros((3,), dtype=np.float32)
        self.root_quat_world = np.zeros((1,4), dtype=np.float32)
        self.root_quat_world_tmp = np.zeros((1,4), dtype=np.float32)
        self.root_rpy = np.zeros((1,3), dtype=np.float32)
        self.base_lin_vel = np.zeros((1,3), dtype=np.float32)
        self.base_ang_vel = np.zeros((1,3),dtype=np.float32)
        self.projected_gravity = np.zeros((1,3), dtype=np.float32)
        self.dof_pos = np.zeros((1, self.num_action), dtype=np.float32)
        self.dof_vel = np.zeros((1, self.num_action), dtype=np.float32)
        self.marker_pos = None
        self.action = np.zeros((1, self.num_action), dtype=np.float32)
        self.last_action = np.zeros((1, self.num_action), dtype=np.float32)
        self.act = np.zeros((1, self.num_action), dtype=np.float32)
        self.foot_contact_forces_w = None

        self.teleop_dof_pos = np.zeros((self.num_action), dtype=np.float64)
        self.teleop_dof_pos_tmp = np.zeros((self.num_action), dtype=np.float64)
        self.teleop_quat = np.array([0, 0, 0, 1], dtype=np.float64)
        self.teleop_quat_tmp = np.array([0, 0, 0, 1], dtype=np.float64)

    def update_obs(self):
        self.get_state()

        return {
            'actions': self.action,
            'root_quat': self.root_quat,
            'root_rpy': self.root_rpy,
            'base_ang_vel': self.base_ang_vel,
            'base_lin_vel': self.base_lin_vel,
            'dof_pos': self.dof_pos,
            'dof_vel': self.dof_vel,
            'projected_gravity': self.projected_gravity,
        }
    
    def update_marker_pos(self, marker_pos):
        self.marker_pos = marker_pos.squeeze(0)
    
    def check_termination(self):
        raise NotImplementedError

    def calibrate(self, refresh, init_ref_dof_pos=None):
        raise NotImplementedError
    
    def apply_action(self, action):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def fk(self):
        if self.kinematic is None:
            raise ValueError("Kinematic model is not initialized")
        fk_info, fk_info_tensor = self.kinematic.forward(
            joint_pos=self.dof_pos,
            base_pos=self.root_trans,
            base_quat=self.root_quat,
            joint_vel=self.dof_vel,
            base_lin_vel=self.base_lin_vel, 
            base_ang_vel=self.base_ang_vel
        )
        return fk_info, fk_info_tensor
    
    # =========================== Teleop Functions ===========================
    def fk_teleop(self):
        if self.kinematic is None:
            raise ValueError("Kinematic model is not initialized")
        
        fk_info, fk_info_tensor = self.kinematic.forward(
            joint_pos=self.teleop_dof_pos_tmp,
            base_pos=np.zeros((3), dtype=np.float64),
            base_quat=self.teleop_quat,
            joint_vel=None,
            base_lin_vel=None, 
            base_ang_vel=None
        )
        return fk_info, fk_info_tensor
    
    def poll(self, cb=None):
        t = time.time()
        try:
            while True:
                timeout = 0.01
                rfds, wfds, efds = select.select([self.lc.fileno()], [], [], timeout)
                if rfds:
                    self.lc.handle()
                else:
                    continue
        except KeyboardInterrupt:
            pass

    def spin(self):
        self.run_thread = threading.Thread(target=self.poll, daemon=True)
        self.run_thread.start()

    def close(self):
        self.lc.unsubscribe(self.joint_state_subscriber)

    def connected(self):
        return self.firstReceiveAlarm
    
    def _teleop_state_handler(self, channel, data):
        msg = camera_reference_data_lcmt.decode(data)
        if not self.firstReceiveAlarm:
            self.firstReceiveAlarm = True
            logger.info('Teleop Information Received!')
        
        self.teleop_dof_pos_tmp = np.array(msg.cam_ref_dof_pos)
        self.teleop_quat_tmp = np.roll(msg.root_rot, -1)  # wxyz -> xyzw
    
    def reset_teleop(self):
        rot = sRot.from_quat(self.teleop_quat_tmp.copy())
        # yaw_only
        euler = rot.as_euler("xyz")
        euler[0] = 0.0
        euler[1] = 0.0
        rot = sRot.from_euler("xyz", euler)

        self._base_rot = rot
        self._init_teleop = True
        self._base_pos = np.array([0, 0, -0.78])

    def align_quat(self, quat: np.ndarray) -> np.ndarray:
        current = sRot.from_quat(quat)
        aligned = self._base_rot.inv() * current
        return aligned.as_quat()
    
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
        return rel.reshape(pos.shape).astype(np.float32)
