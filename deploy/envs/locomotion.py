import torch
import numpy as np
from envs.base_env import BaseEnv
from loguru import logger

class Locomotion(BaseEnv):
    def __init__(self, config):
        super().__init__(config)
        self.device = self.cfg.get('device', 'cpu')
        
        # Locomotion command variables
        self.command_lin_vel = np.zeros((1, 2), dtype=np.float32)  # [x, y] velocity
        self.command_ang_vel = np.zeros((1, 1), dtype=np.float32)  # yaw velocity
        
        # Command ranges
        self.max_lin_vel = 2.0  # m/s
        self.max_ang_vel = 2.0  # rad/s
        
        logger.info("Locomotion environment initialized")

    def _reset_envs(self, refresh):
        super()._reset_envs(refresh)
        # Reset commands to zero
        self.command_lin_vel *= 0
        self.command_ang_vel *= 0
        logger.info("Locomotion environment reset")

    def _update_obs(self):
        super()._update_obs()
        # Update command velocities (these could be set externally or from joystick input)
        self._update_commands()

    def _update_commands(self):
        # This method can be overridden to get commands from external sources
        # For now, keep commands as they are (can be set externally)
        pass

    def set_command_velocity(self, lin_vel_x, lin_vel_y, ang_vel_z):
        """Set locomotion command velocities"""
        self.command_lin_vel[0, 0] = np.clip(lin_vel_x, -self.max_lin_vel, self.max_lin_vel)
        self.command_lin_vel[0, 1] = np.clip(lin_vel_y, -self.max_lin_vel, self.max_lin_vel)
        self.command_ang_vel[0, 0] = np.clip(ang_vel_z, -self.max_ang_vel, self.max_ang_vel)

    def _get_obs_command_lin_vel(self):
        return self.command_lin_vel
    
    def _get_obs_command_ang_vel(self):
        return self.command_ang_vel

    def _check_termination(self):
        # Check basic termination conditions
        hard_reset = self.simulator.check_termination()
        
        if hard_reset:
            logger.warning("Locomotion terminated due to failure")
            self._reset_envs(True)
            self.compute_observation()
            
        return hard_reset 