"""
Agent that manages switching between locomotion and mimic policies with smooth interpolation.
Inspired by RoboJuDo's RlLocoMimicPipeline.
"""

import time
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch
from loguru import logger

from agents.base_agent import BaseAgent
from utils.dataset import MosaicModelMeta


class InterpState(Enum):
    """Interpolation state machine."""

    IDLE = auto()
    START = auto()
    IN_PROGRESS = auto()
    END = auto()


class PolicyInterpManager:
    """
    Manages smooth interpolation between locomotion and mimic policies.
    """

    # Interpolation durations in steps: [start_delay, interpolation_steps, end_delay]
    DURATIONS_LOCO_MIMIC = [0, 75, 25]  # locomotion -> mimic
    DURATIONS_MIMIC_LOCO = [25, 75, 0]  # mimic -> locomotion

    def __init__(self, env):
        self.env = env
        
        # Interpolation state
        self.interp_state = InterpState.IDLE
        self.interp_timestep = 0
        self.interp_durations = [20, 40, 20]
        self.interp_callback_start = None
        self.interp_callback_end = None
        
        # Target positions
        self.loco_dof_pos = self.env.simulator.default_angles[self.env.simulator.active_dof_idx].copy()
        self.override_dof_pos = self.loco_dof_pos.copy()
        
        # Interpolation data
        self.interp_start_pos: Optional[np.ndarray] = None
        self.interp_target_pos: Optional[np.ndarray] = None
        self.interp_get_target_pos = None
        
        # Timer for delayed actions
        self.pending_actions = []

    def _interpolate_init(self, get_target_pos, durations, callback_start=None, callback_end=None):
        """Initialize interpolation process."""
        self.interp_get_target_pos = get_target_pos
        self.interp_durations = durations
        self.interp_callback_start = callback_start
        self.interp_callback_end = callback_end
        
        self.interp_state = InterpState.START
        self.interp_timestep = 0
        
        # Schedule callbacks
        if durations[0] == 0:
            self._interpolate_start()
        else:
            self.pending_actions.append(("start", durations[0]))
        
        total_duration = sum(durations)
        self.pending_actions.append(("end", total_duration + 1))

    def _interpolate_start(self):
        """Start interpolation."""
        if self.interp_state != InterpState.START:
            return
        
        if self.interp_callback_start is not None:
            self.interp_callback_start()
            self.interp_callback_start = None
        
        self.interp_start_pos = self.env.dof_pos.squeeze().copy()
        self.interp_target_pos = self.interp_get_target_pos()
        self.interp_timestep = 0
        self.interp_state = InterpState.IN_PROGRESS
        
        logger.debug("Interpolation started")

    def _interpolate_end(self):
        """End interpolation."""
        if self.interp_state != InterpState.END:
            return
        
        self.override_dof_pos = self.interp_target_pos.copy()
        
        if self.interp_callback_end is not None:
            self.interp_callback_end()
            self.interp_callback_end = None
        
        self.interp_state = InterpState.IDLE
        logger.debug("Interpolation ended")

    def _interpolate_step(self):
        """Perform one interpolation step."""
        if self.interp_state != InterpState.IN_PROGRESS:
            return
        
        progress = self.interp_timestep / self.interp_durations[1]
        alpha = min(progress, 1.0)
        self.override_dof_pos = (1 - alpha) * self.interp_start_pos + alpha * self.interp_target_pos
        
        if self.interp_timestep < self.interp_durations[1]:
            self.interp_timestep += 1
        else:
            self.interp_state = InterpState.END

    def switch_to_loco(self):
        """Switch from mimic to locomotion policy."""
        if self.env.policy_mode == "locomotion" and self.interp_state == InterpState.IDLE:
            logger.warning("Already in locomotion policy")
            return
        
        logger.info("Switching to locomotion policy")
        
        self._interpolate_init(
            get_target_pos=lambda: self.loco_dof_pos,
            durations=self.DURATIONS_MIMIC_LOCO,
            callback_start=lambda: self.env.set_policy_mode("locomotion"),
        )

    def switch_to_mimic(self):
        """Switch from locomotion to mimic policy."""
        if self.env.policy_mode == "mimic":
            logger.warning("Already in mimic policy")
            return
        
        if self.env.mimic_model_meta is None:
            logger.error("Mimic policy not configured")
            return
        
        logger.info("Switching to mimic policy")
        
        # Reset motion loader to start from first frame
        self.env.motion_loader.reset()
        self.env.motion_finished = False
        self.env.time_step = 0.0
        
        # Get initial mimic position from motion data
        # Note: motion_loader.joint_pos is in mimic policy DoF order,
        # need to convert to simulator DoF order for interpolation
        def get_mimic_init_pos():
            # Get joint positions from motion data (mimic policy DoF order)
            mimic_joint_pos = self.env.motion_loader.joint_pos.copy()
            # Convert to simulator DoF order
            sim_joint_pos = self.env._mimic_to_sim(mimic_joint_pos)
            logger.debug(f"Mimic init pos: policy DoF -> simulator DoF")
            logger.debug(f"Target position range: [{sim_joint_pos.min():.3f}, {sim_joint_pos.max():.3f}]")
            return sim_joint_pos
        
        self._interpolate_init(
            get_target_pos=get_mimic_init_pos,
            durations=self.DURATIONS_LOCO_MIMIC,
            callback_end=lambda: self.env.set_policy_mode("mimic"),
        )

    def step(self):
        """Update interpolation state (call every step)."""
        # Process pending actions
        new_pending = []
        for action_type, delay in self.pending_actions:
            if delay <= 0:
                if action_type == "start":
                    self._interpolate_start()
                elif action_type == "end":
                    self._interpolate_end()
            else:
                new_pending.append((action_type, delay - 1))
        self.pending_actions = new_pending
        
        # Update interpolation
        self._interpolate_step()


class LocoMimicAgent(BaseAgent):
    """
    Agent that manages switching between locomotion and mimic policies.
    Supports smooth interpolation between different policy states.
    """

    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = self.config.get("device", "cpu")
        self.time = 0.0
        
        # Load policies
        self._load_locomotion_policy()
        self._load_mimic_policy()
        
        # Create interpolation manager
        self.interp_manager = PolicyInterpManager(self.env)
        
        # Command handling
        self.enable_keyboard_commands = self.config.get("enable_keyboard_commands", True)
        
        logger.info("LocoMimicAgent initialized with locomotion and mimic policies")

    def _load_locomotion_policy(self):
        """Load locomotion policy (TorchScript)."""
        loco_ckpt = self.config.locomotion_checkpoint
        self.loco_policy = torch.jit.load(loco_ckpt).to(self.device)
        self.loco_policy.eval()
        logger.info(f"Loaded locomotion policy from {loco_ckpt}")

    def _load_mimic_policy(self):
        """Load mimic policy (ONNX) and configure environment."""
        import onnxruntime as ort
        
        mimic_ckpt = self.config.mimic_checkpoint
        self.mimic_policy = ort.InferenceSession(mimic_ckpt)
        logger.info(f"Loaded mimic policy from {mimic_ckpt}")
        
        # Extract and configure metadata
        model_meta = MosaicModelMeta.from_onnx_session(self.mimic_policy)
        self.env.configure_mimic_from_modelmeta(model_meta)
        logger.info(
            "Mimic policy metadata configured: {} joints, anchor body: {}",
            len(model_meta.joint_names),
            model_meta.anchor_body_name,
        )

    def _get_policy_action(self, obs_buf_dict):
        """Get action from current policy."""
        if self.env.policy_mode == "locomotion":
            actor_obs = torch.from_numpy(obs_buf_dict["actor_obs"]).float().to(self.device)
            with torch.no_grad():
                action = self.loco_policy(actor_obs).cpu().numpy()
        elif self.env.policy_mode == "mimic":
            inputs = {key: obs_buf_dict[key].astype(np.float32) for key in obs_buf_dict}
            ort_outputs = self.mimic_policy.run(None, inputs)
            action = ort_outputs[0]
        else:
            raise ValueError(f"Unknown policy mode: {self.env.policy_mode}")
        
        return action

    def _apply_override_dof_pos(self, action):
        """
        During interpolation, pass override positions to environment.
        The environment will handle the DoF composition.
        """
        if self.interp_manager.interp_state == InterpState.IDLE:
            self.env.override_dof_pos = None
            return action
        
        # Pass override positions to environment for proper DoF handling
        self.env.override_dof_pos = self.interp_manager.override_dof_pos.copy()
        
        return action

    def _handle_commands(self):
        """Handle user commands for policy switching."""
        if not self.enable_keyboard_commands:
            return
        
        # Check for pending policy switch from environment (teleop)
        pending_switch = self.env.get_pending_policy_switch()
        if pending_switch is not None:
            if pending_switch == "locomotion":
                self.interp_manager.switch_to_loco()
            elif pending_switch == "mimic":
                self.interp_manager.switch_to_mimic()

    def run(self):
        """Main agent loop."""
        self.time = time.time()
        obs_buf_dict = self.env.reset()
        
        step_count = 0
        prev_policy_mode = self.env.policy_mode
        
        while True:
            # Get action from current policy
            action = self._get_policy_action(obs_buf_dict)
            
            # Apply DoF override during interpolation
            if action.ndim == 1:
                action = action[None, :]
            action = self._apply_override_dof_pos(action)
            
            # Step environment
            obs_buf_dict = self.env.step(action.squeeze())
            
            # Update interpolation
            self.interp_manager.step()
            
            # Handle commands (e.g., policy switching)
            self._handle_commands()
            
            # If policy mode changed, update obs_buf_dict with new observation
            if self.env.policy_mode != prev_policy_mode:
                logger.debug(f"Policy mode changed: {prev_policy_mode} -> {self.env.policy_mode}")
                prev_policy_mode = self.env.policy_mode
                # Observation was recomputed in set_policy_mode(), get the updated dict
                obs_buf_dict = self.env.obs_buf_dict
            
            # Timing control for real robot
            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Step timeout: {time_until_next_step:.4f}s")
            
            self.time = time.time()
            step_count += 1
            
            # Auto-switch demo (optional): switch to mimic after N steps, then back
            if self.config.get("auto_switch_demo", False):
                if step_count == 100:
                    self.interp_manager.switch_to_mimic()
                elif step_count == 500:
                    self.interp_manager.switch_to_loco()
                    
            # Check if mimic motion is finished
            if self.env.policy_mode == "mimic" and self.env.motion_finished:
                logger.info("Mimic motion finished, switching back to locomotion")
                self.interp_manager.switch_to_loco()

    def switch_to_locomotion(self):
        """Public API to switch to locomotion."""
        self.interp_manager.switch_to_loco()

    def switch_to_mimic(self):
        """Public API to switch to mimic."""
        self.interp_manager.switch_to_mimic()

