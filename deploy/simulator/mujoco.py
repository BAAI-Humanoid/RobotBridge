import os
import time
import lcm
import threading
import copy
import numpy as np
import torch
from simulator.base_sim import BaseSim
import mujoco.viewer
import mujoco
import math
from scipy.spatial.transform import Rotation as R
from utils.helpers import get_gravity, get_rpy, quaternion_to_euler_array
from utils.motion_lib.rotations import calc_heading_quat_inv, my_quat_rotate
from loguru import logger
from scipy.spatial.transform import Rotation as sRot
# from pynput import keyboard

DESIRED_BODY_INDICES = [0, 2, 4, 6, 8, 10, 12, 15, 17, 19, 22, 24, 26, 29]

class Mujoco(BaseSim):
    is_real = False
    def __init__(self, config):
        super().__init__(config)
        self.marker = self.cfg.get('marker', False)
        self.real_start_time = None
        self.sim_start_time = 0

        logger.info(f'Visualization Marker: {self.marker}')
        self.target_dof_pos = None  # Initialize target_dof_pos
        if self.cfg.control.viewer:
            self._load_viewer()

        self._init_communication()
        self.spin()
        if getattr(self.cfg.control, "use_teleop", False):
            while True:
                if self.connected:
                    print("============================== init sim done ==============================")
                    break
        else:
            self.firstReceiveAlarm = True

        # Use keyboard listener to toggle pause state
        self.paused = False
        # self.listener = keyboard.Listener(on_press=self._on_press_fallback)
        # self.listener.start()

    def _setup(self):
        super()._setup()
        self.lc = lcm.LCM('udpm://239.255.76.67:7667?ttl=255')

    def _load_asset(self):
        super()._load_asset()
        xml_path = os.path.join(self.cfg.asset.asset_root, self.cfg.asset.asset_file)
        
        self.mujoco_model = mujoco.MjModel.from_xml_path(xml_path)
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        self.mujoco_model.opt.timestep = self.low_dt
        print("mujoco time step : ", self.mujoco_model.opt.timestep)
        
        self.default_qpos = self.mujoco_data.qpos.copy()
        self.default_qvel = self.mujoco_data.qvel.copy()
        
        # Compute frozen DOF indices (complement of active DOFs)
        self.frozen_dof_idx = np.array([i for i in range(self.num_dof) if i not in self.active_dof_idx], dtype=np.int32)
        foot_body_names = getattr(self.cfg.asset, "foot_body_names", None)
        if not foot_body_names:
            foot_body_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
        self.foot_body_names = list(foot_body_names)
        self.foot_body_ids = []
        for name in self.foot_body_names:
            body_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                logger.warning(f"[Mujoco] foot body not found: {name}")
                continue
            self.foot_body_ids.append(body_id)
        self.foot_body_ids = np.asarray(self.foot_body_ids, dtype=np.int32)
        self.foot_contact_forces_w = np.zeros((len(self.foot_body_ids), 3), dtype=np.float32)

    def _init_communication(self):
        self.firstReceiveAlarm = False
        self.firstReceiveOdometer = False
        self._init_time = time.time()
        
        self.teleop_state_subscriber = self.lc.subscribe('camera_reference_data', self._teleop_state_handler)

    def _on_press_fallback(self, key):
        try:
            if key == keyboard.Key.space:
                self.paused = not self.paused
                print(f"\n[GLOBAL PAUSE] Status: {self.paused}", flush=True)

            if hasattr(key, 'char') and key.char == 'c':
                cam = self.viewer.cam
                print("\n" + "="*30)
                print("Current Camera Parameters:")
                print(f"self.viewer.cam.lookat[:] = np.array([{cam.lookat[0]:.8f}, {cam.lookat[1]:.8f}, {cam.lookat[2]:.8f}])")
                print(f"self.viewer.cam.distance = {cam.distance:.4f}")
                print(f"self.viewer.cam.azimuth = {cam.azimuth:.4f}")
                print(f"self.viewer.cam.elevation = {cam.elevation:.4f}")
                print("="*30 + "\n")
        except Exception as e:
            print(f"Error in keyboard listener: {e}")

    def _load_viewer(self):
        self.viewer = mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data)
        
        # 1. Fixed Mode: Static camera view
        self.viewer.cam.lookat[:] = np.array([0.0,0.0,0.8])
        self.viewer.cam.distance = 2.5        
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -10   
        
        # 2. Tracking Mode: Camera follows the robot body (uncomment to use)
        # self.viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        
        # body_name = "torso_link"
        # body_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        # self.viewer.cam.trackbodyid = body_id
        
        # self.viewer.cam.lookat[:] = np.array([0.0, 0.0, 0.0]) 
        # self.viewer.cam.distance = 3       
        # self.viewer.cam.azimuth = 180
        # self.viewer.cam.elevation = 0
        
        self.marker_pos = None

    def render(self):
        if self.marker_pos is not None:
            self.viewer.user_scn.ngeom = 0
            for i in range(self.marker_pos.shape[0]):
                # Ensure pos is a 3D vector
                pos = self.marker_pos[i].flatten()[:3]  # Take first 3 elements and flatten
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.03, 0, 0], dtype=np.float64),
                    pos=pos.astype(np.float64),
                    mat=np.eye(3).flatten().astype(np.float64),
                    rgba=np.array([1,0,0,1], dtype=np.float32)
                )
            self.viewer.user_scn.ngeom=self.marker_pos.shape[0]
        self.viewer.sync()

    def get_state(self):
        data = self.mujoco_data
        self.root_quat = data.qpos.astype(np.double)[3:7][[1,2,3,0]]       # WXYZ to XYZW
        self.root_quat_world = data.qpos.astype(np.double)[3:7][[1,2,3,0]] # world quat in mujoco WXYZ to XYZW
        r = R.from_quat(self.root_quat)  # R.from_quat: need xyzw
        self.root_rpy = quaternion_to_euler_array(self.root_quat) # need xyzw
        self.root_rpy[self.root_rpy > math.pi] -= 2 * math.pi
        
        self.root_trans = data.qpos[:3].astype(np.double)
        self.root_trans_world = data.qpos[:3].astype(np.double)

        self.projected_gravity = r.apply(np.array([0., 0., -1.]), inverse=True).astype(np.double)
        lin_vel_world = data.qvel.astype(np.double)[0:3]
        self.base_lin_vel = r.apply(lin_vel_world, inverse=True).astype(np.float32)
        self.base_ang_vel = data.qvel.astype(np.double)[3:6] # local
        
        # Get positions and velocities for active DOFs only
        all_dof_pos = data.qpos[7:].astype(np.double)
        all_dof_vel = data.qvel[6:].astype(np.double)
        self.dof_pos = all_dof_pos[self.active_dof_idx]
        self.dof_vel = all_dof_vel[self.active_dof_idx]

        if self.cfg.control.update_with_fk:
            fk_info, fk_info_tensor = self.fk()
            self.torso_quat = fk_info[self.torso_name]['quat']
            self.torso_trans = fk_info[self.torso_name]['pos']
            self.robot_fk_info = fk_info_tensor

        if hasattr(self.cfg.control, 'use_teleop'):
            if self.cfg.control.use_teleop:
                idx_r2p = [
                    0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28
                ]
                self.teleop_dof_pos = self.teleop_dof_pos_tmp.copy()[idx_r2p]
                self.teleop_quat = self.align_quat(self.teleop_quat_tmp.copy())
                if self.cfg.control.update_with_fk:
                    fk_info, fk_info_tensor = self.fk_teleop()
                    self.teleop_quat = fk_info[self.torso_name]['quat']
                    teleop_body_pos_w = fk_info_tensor[DESIRED_BODY_INDICES, :3]
                    self.teleop_body_pos_w_aligned = self.align_pos_batch(teleop_body_pos_w)

        if self.foot_body_ids.size > 0:
            forces_body = data.cfrc_ext[self.foot_body_ids, :3].astype(np.float32)
            xmat = data.xmat[self.foot_body_ids].reshape(-1, 3, 3)
            self.foot_contact_forces_w = np.einsum("nij,nj->ni", xmat, forces_body)
        else:
            self.foot_contact_forces_w = np.zeros((0, 3), dtype=np.float32)

    def apply_action(self, action):
        # 1. record the timestamp when first enter simulator
        if self.real_start_time is None:
            self.real_start_time = time.perf_counter()
            self.sim_start_time = self.mujoco_data.time

        self.act = action.copy()
        is_mosaic = getattr(self.cfg.control, 'is_mosaic', False)
        
        # 2. compute target pos
        if is_mosaic:
            tgt_dof_pos = action
        else:
            current_default_angles = self.default_angles[self.active_dof_idx].copy()
            use_residual = getattr(self.cfg.control, 'use_residual', False)
            if use_residual and self.ref_dof_pos is not None:
                if hasattr(self.cfg.control, 'residual_joint_indices') and self.cfg.control.residual_joint_indices is not None:
                    for idx in self.cfg.control.residual_joint_indices:
                        current_default_angles[idx] = self.ref_dof_pos[idx]
            self.dof_tracking_init_pos = current_default_angles
            tgt_dof_pos = current_default_angles + action * self.cfg.control.action_scale

        # 3. physical stepping
        torque_limit = np.array(self.cfg.control.torque_clip_value, dtype=np.float32)
        
        for _ in range(self.decimation):
            while self.paused:
                if not self.viewer.is_running():
                    break
                self.render()      # Keep rendering the screen, otherwise the window will freeze
                time.sleep(0.01)

            all_dof_pos = self.mujoco_data.qpos[7:].astype(np.float32)
            all_dof_vel = self.mujoco_data.qvel[6:].astype(np.float32)

            torque_active = (tgt_dof_pos - all_dof_pos[self.active_dof_idx]) * self.kps[self.active_dof_idx] \
                            - all_dof_vel[self.active_dof_idx] * self.kds[self.active_dof_idx]
            torque_active = np.clip(torque_active, -torque_limit, torque_limit)

            if len(self.frozen_dof_idx) > 0:
                torque_all = np.zeros(self.num_dof, dtype=np.float32)
                torque_all[self.active_dof_idx] = torque_active
                torque_frozen = (self.default_angles[self.frozen_dof_idx] - all_dof_pos[self.frozen_dof_idx]) * self.kps[self.frozen_dof_idx] \
                                - all_dof_vel[self.frozen_dof_idx] * self.kds[self.frozen_dof_idx]
                torque_all[self.frozen_dof_idx] = torque_frozen
                self.mujoco_data.ctrl[:] = torque_all
            else:
                self.mujoco_data.ctrl[:] = torque_active

            mujoco.mj_step(self.mujoco_model, self.mujoco_data)

        # 4. rendering
        if self.cfg.control.viewer:
            self.render()

        # 5. (Optional) sync with real time
        is_real_time = getattr(self.cfg.control, 'real_time', False)
        if is_real_time:
            sim_elapsed = self.mujoco_data.time - self.sim_start_time
            real_elapsed = time.perf_counter() - self.real_start_time
            
            # sync with real time
            if sim_elapsed > real_elapsed:
                time_to_wait = sim_elapsed - real_elapsed
                if time_to_wait > 0.0005: 
                    time.sleep(time_to_wait)

    def calibrate(self, refresh, init_ref_dof_pos=None):
        if refresh:
            default_qpos = self.default_qpos.copy()
            default_qvel = self.default_qvel

            # Use init_ref_dof_pos if provided, otherwise use default_angles
            # Important: default_qpos[7:] needs ALL joint angles (num_dof), not just active ones (num_action)
            if init_ref_dof_pos is not None:
                # Start with all default angles
                current_default_angles = self.default_angles.copy()
                use_residual = getattr(self.cfg.control, 'use_residual', False)    
                if use_residual and init_ref_dof_pos is not None:
                    # Get residual joint indices from config
                    if hasattr(self.cfg.control, 'residual_joint_indices') and self.cfg.control.residual_joint_indices is not None:
                        residual_joint_indices = self.cfg.control.residual_joint_indices
                        # Flatten init_ref_dof_pos if it's 2D
                        ref_dof_pos_flat = init_ref_dof_pos.flatten() if init_ref_dof_pos.ndim > 1 else init_ref_dof_pos
                        for idx in residual_joint_indices:
                            if idx < len(ref_dof_pos_flat) and idx < len(current_default_angles):
                                current_default_angles[idx] = ref_dof_pos_flat[idx]
                default_qpos[7:] = current_default_angles
            else:
                # Use complete default_angles for all joints
                default_qpos[7:] = self.default_angles

            logger.info(f"Resetting envs with ref_dof_pos={[f'{x:.3f}' for x in default_qpos[7:].tolist()]}")
            self.mujoco_data.qpos[:] = default_qpos.copy()
            self.mujoco_data.qvel[:] = default_qvel
            self.mujoco_data.ctrl[:] = 0
        else:
            self.get_state()
            cur_dof_pos = self.dof_pos
            final_goal = np.zeros_like(self.default_angles[self.active_dof_idx])
            default_pos = self.default_angles[self.active_dof_idx]
            target = cur_dof_pos - default_pos
            target_seq=[]
            while np.max(np.abs(target-final_goal)) > 0.01:
                target-=np.clip((target-final_goal), -0.05, 0.05)
                target_seq += [copy.deepcopy(target)]
            for tgt in target_seq:
                next_tgt = tgt/self.cfg.control.action_scale
                self.apply_action(next_tgt[None,])
                self.get_state()
            logger.info(f'Simulation Done!')
            while True:
                self.apply_action(np.zeros((1, self.num_action)))
        self.reset_teleop()
    
    def check_termination(self):
        return abs(self.root_rpy[0]) > 1.2 or abs(self.root_rpy[1]) > 1.2
        # return False
