import time

import torch
from loguru import logger

from agents.base_agent import BaseAgent


class TwistAgent(BaseAgent):
    def __init__(self, config, env):
        self.config = config
        self.time = 0
        self.print_cnt = 0
        self.env = env
        self.device = self.config.get("device", "cpu")

        self.load_policy()

    def _inference(self, obs_buf_dict):
        obs_numpy = obs_buf_dict["obs"]
        obs_tensor = torch.from_numpy(obs_numpy).float().to(self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)

        with torch.no_grad():
            action_tensor = self.policy(obs_tensor)

        return action_tensor.detach().cpu().numpy().squeeze()

    def run(self):
        obs_buf_dict = self.env.reset()
        self.time = time.time()
        while True:
            action = self._inference(obs_buf_dict)
            obs_buf_dict = self.env.step(action)

            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Time until next step is negative: {time_until_next_step}")
            self.time = time.time()

    def run_eval(self):
        obs_buf_dict = self.env.reset()
        self.time = time.time()
        while True:
            action = self._inference(obs_buf_dict)
            obs_buf_dict = self.env.step(action)

            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Time until next step is negative: {time_until_next_step}")
            self.time = time.time()

            if getattr(self.env.motion_loader, "motion_finished", False):
                obs_buf_dict = self.env.reset()
