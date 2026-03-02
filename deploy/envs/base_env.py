import os
import torch
import time
import joblib
import numpy as np
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from utils.history_handler import HistoryHandler
from utils.motion_lib.motion_lib_robot import MotionLibRobot
from utils.helpers import parse_observation
from loguru import logger

class BaseEnv:
    def __init__(self, config):
        self.cfg = config

        self.simulator = instantiate(self.cfg.simulator)
        self.device = self.cfg.get('device', 'cpu')
        self._init_buffer()

    def _init_buffer(self):
        self.num_dof = self.simulator.num_dof
        self.num_action = self.simulator.num_action
        
        self.root_trans = np.zeros((1,3),dtype=np.float32)
        self.root_trans_world = np.zeros((1,3),dtype=np.float32)
        self.root_quat = np.zeros((1, 4), dtype=np.float32)
        self.root_quat_world = np.zeros((1,4), dtype=np.float32)
        self.root_quat_world[0] = [0., 0., 0., 1.]
        self.root_rpy = np.zeros((1,3), dtype=np.float32)
        self.base_lin_vel = np.zeros((1,3), dtype=np.float32)
        self.base_ang_vel = np.zeros((1,3),dtype=np.float32)
        self.projected_gravity = np.zeros((1,3), dtype=np.float32)
        self.dof_pos = np.zeros((1, self.num_action), dtype=np.float32)
        self.dof_vel = np.zeros((1, self.num_action), dtype=np.float32)
        self.action = np.zeros((1, self.num_action), dtype=np.float32)
        
        self.episode_length_buf = np.zeros((1,), dtype=np.float64)
        self.history_handler = HistoryHandler(self.cfg.obs.obs_auxiliary, self.cfg.obs.obs_dims)

        self.init_ref_dof_pos = None

        self.first_obs_received = False
    
        self.collected_traj = {}

    def reset(self):
        self._reset_envs(refresh=True)
        self.compute_observation()
        return self.obs_buf_dict

    def _reset_envs(self, refresh):
        self.episode_length_buf*=0
        self.action *= 0
        self.history_handler.reset()
        self.collected_traj = {}
        # Pass ref_dof_pos if it exists, otherwise pass None
        ref_dof_pos = self.init_ref_dof_pos
        # logger.info(f"Resetting envs with ref_dof_pos={[f'{x:.3f}' for x in ref_dof_pos.tolist()]}")
        self.simulator.calibrate(refresh, ref_dof_pos)

    def compute_observation(self):
        self._update_obs()
        self._assemble_observations()
        self._post_compute_observations_callback()

    def step(self, action):
        self._pre_physics_step(action)
        self._physics_step()
        self._post_physics_step()
        return self.obs_buf_dict

    def _pre_physics_step(self, action):
        clip_action_limit = self.cfg.control.action_clip_value
        action = np.asarray(action, dtype=np.float32)
        if clip_action_limit is None or clip_action_limit <= 0:
            self.action = action
        else:
            self.action = np.clip(action, -clip_action_limit, clip_action_limit)

    def _physics_step(self):
        self.simulator.apply_action(self.action)
    
    def _post_physics_step(self):
        self.episode_length_buf+=1
        self.compute_observation()
        self._check_termination()

    def _update_obs(self):
        self.prop_obs_dict = self.simulator.update_obs()
        if not self.first_obs_received:
            logger.info(f'Receive observation: {list(self.prop_obs_dict.keys())}')
            self.first_obs_received = True
        for k in self.prop_obs_dict:
            if k in self.__dict__:
                self.__dict__[k] = self.prop_obs_dict[k][None,]
    
    def _save_collected_traj(self, complete):
        if self.cfg.collect_dataset and self.collected_traj:
            save_dir = HydraConfig.get().runtime.output_dir
            traj_files = [f for f in os.listdir(save_dir) if f .startswith('traj_') and f.endswith('.pkl')]
            indices = [int(f.split('_')[1].split('.')[0]) for f in traj_files]
            next_index = max(indices)+1 if indices else 0
            file_name = f'traj_{next_index}'
            if complete:
                file_name+='_complete.pkl'
                logger.info('You have collected a complete trajectory!')
            else:
                file_name+="_terminated.pkl"
                logger.warning('The robot suffers from abrupt termination. The trajectory is incomplete!')
            file_path = os.path.join(save_dir, file_name)
            joblib.dump(self.collected_traj, file_path)
            logger.info(f'Trajectory {next_index} saved to {file_path}. There are {len(self.collected_traj)} frames in total. ')
            

    def _check_termination(self):
        hard_reset = self.simulator.check_termination()
        if hard_reset:
            self._save_collected_traj(False)
            self._reset_envs(True)
            self.compute_observation()

    def _assemble_observations(self):
        """
        Enhanced observation assembly using URCIRobotResidual's separated approach
        """
        
        self._update_obs_without_history()
        self._update_obs_for_history()

        self.obs_buf_dict = {}
        obs_cfg = self.cfg.obs
        for obs_key, obs_config in obs_cfg.obs_dict.items():
            obs_keys = sorted(obs_config)
            self.obs_buf_dict[obs_key] = np.concatenate(
                [self.obs_buf_dict_raw[obs_key][k] for k in obs_keys], axis=-1
            )
        
        # Collect trajectory data if enabled
        if self.cfg.collect_dataset:
            ts = int(self.episode_length_buf[0])
            self.collected_traj[ts] = self.obs_buf_dict_raw.copy()

    def _update_obs_without_history(self):
        """
        Process current observations without history
        Similar to URCIRobotResidual.UpdateObsWoHistory()
        """
        self.obs_buf_dict_raw = {}
        obs_cfg = self.cfg.obs
        
        for obs_key, obs_config in obs_cfg.obs_dict.items():
            self.obs_buf_dict_raw[obs_key] = {}
            parse_observation(
                self,
                obs_config,
                self.obs_buf_dict_raw[obs_key],
                obs_cfg.obs_scales,
            )
    
    def _update_obs_for_history(self):
        """
        Process history observations separately
        Similar to URCIRobotResidual.UpdateObsForHistory()
        """
        hist_cfg_obs = self.cfg.obs
        self.hist_obs_dict = {}
        
        # Get observation types that need history tracking
        history_obs_list = self.history_handler.history.keys()
        parse_observation(
            self,
            history_obs_list,
            self.hist_obs_dict,
            hist_cfg_obs.obs_scales,
        )

    def _post_compute_observations_callback(self):
        clip_obs_limit = self.cfg.control.obs_clip_value
        for obs_key in self.obs_buf_dict:
            self.obs_buf_dict[obs_key] = np.clip(self.obs_buf_dict[obs_key], -clip_obs_limit, clip_obs_limit)

    # The function name below should not be modified as the observation is in alphabetical order!!!

    def _get_obs_root_quat(self):
        return self.root_quat
    
    def _get_obs_root_vel(self):
        return self.base_lin_vel
    
    def _get_obs_base_ang_vel(self):
        return self.base_ang_vel
    
    # def _get_obs_dof_pos(self):
    #     return self.dof_pos - self.simulator.dof_tracking_init_pos
    
    def _get_obs_dof_pos_relative(self):
        return self.dof_pos - self.simulator.default_angles[self.simulator.active_dof_idx]
    
    def _get_obs_dof_pos(self):
        return self.dof_pos

    def _get_obs_dof_vel(self):
        return self.dof_vel
    
    def _get_obs_actions(self):
        return self.action
    
    def _get_obs_projected_gravity(self):
        return self.projected_gravity
    
    def _get_obs_history_actor(self):
        hist_cfg = self.cfg.obs.obs_auxiliary['history_actor']
        hist_arrs=[]

        for key in sorted(hist_cfg.keys()):
            hist_len = hist_cfg[key]
            hist_arr = self.history_handler.query(key)[:hist_len].reshape(1, -1)
            hist_arrs.append(hist_arr)
            # logger.info(f"{key} history tensor shape: {hist_arr.shape}")
        
        return np.concatenate(hist_arrs, axis=-1)
    
    def _get_obs_history_prop(self):
        hist_cfg = self.cfg.obs.obs_auxiliary['history_prop']
        hist_arrs=[]

        for key in sorted(hist_cfg.keys()):
            hist_len = hist_cfg[key]
            hist_arr = self.history_handler.query(key)[:hist_len].reshape(1, -1)
            hist_arrs.append(hist_arr)
            # logger.info(f"{key} history tensor shape: {hist_arr.shape}")
        
        return np.concatenate(hist_arrs, axis=-1)
    
    def _get_obs_history_ref(self):
        hist_cfg = self.cfg.obs.obs_auxiliary['history_ref']
        hist_arrs=[]

        for key in sorted(hist_cfg.keys()):
            hist_len = hist_cfg[key]
            hist_arr = self.history_handler.query(key)[:hist_len].reshape(1, -1)
            hist_arrs.append(hist_arr)
            # logger.info(f"{key} history tensor shape: {hist_arr.shape}")
        
        return np.concatenate(hist_arrs, axis=-1)
        
