import time
import numpy as np
import torch
from loguru import logger


class LevelAgent:
    """
    Simplified Agent using TorchScript LEVEL policy.
    """

    def __init__(self, config, env):
        self.config = config
        self.env = env
        self.device = self.config.get("device", "cpu")

        self.policy = torch.jit.load(self.config.checkpoint).to(self.device)
        self.policy.eval()
        logger.info(f"Loading LEVEL TorchScript policy from {self.config.checkpoint}")

        self.time = 0.0

    def run(self):
        self.time = time.time()
        obs_buf_dict = self.env.reset()

        while True:
            actor_obs = torch.from_numpy(obs_buf_dict["actor_obs"]).float().to(self.device)
            with torch.no_grad():
                action = self.policy(actor_obs).cpu().numpy()

            obs_buf_dict = self.env.step(action)

            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"step timeout {time_until_next_step:.4f}s")
            self.time = time.time()

