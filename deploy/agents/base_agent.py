import time
import numpy as np
import torch
from loguru import logger
import onnxruntime as ort
import os

class BaseAgent:
    def __init__(self, config, env):
        self.config = config
        self.time = 0
        self.print_cnt = 0
        self.env = env
        self.device = self.config.get('device', 'cpu')
        
        self.load_policy()

    def load_onnx_policy(self):
        """Load an ONNX checkpoint."""
        onnx_ckpt_path = self.config.checkpoint
        self.policy = ort.InferenceSession(onnx_ckpt_path)
        logger.info(f'Loading ONNX Checkpoint from {onnx_ckpt_path}')

    def load_jit_policy(self):
        """Load a PyTorch JIT checkpoint."""
        ckpt_path = self.config.checkpoint
        self.policy = torch.jit.load(ckpt_path).to(self.device)
        logger.info(f'Loading Checkpoint from {ckpt_path}')
        
    def load_policy(self):
        """Automatically select the loading method based on the suffix of the checkpoint file"""
        if not hasattr(self.config, 'checkpoint') or not self.config.checkpoint:
            raise ValueError("Config must contain a valid 'checkpoint' path")
        
        ckpt_path = self.config.checkpoint
        file_ext = os.path.splitext(ckpt_path)[1].lower()

        if file_ext == '.onnx':
            self.load_onnx_policy()
        elif file_ext in ['.pt', '.pth', '.jit']:
            self.load_jit_policy()
        else:
            raise ValueError(
                f"Unsupported checkpoint file format: {file_ext}\n"
                f"Supported formats: .onnx, .pt, .pth, .jit\n"
                f"Checkpoint path: {ckpt_path}"
            )
    
    def run(self):
        self.time = time.time()
        obs_buf_dict = self.env.reset()
        while True:
            for key in obs_buf_dict:
                obs_buf_dict[key] = torch.from_numpy(obs_buf_dict[key]).float().to(self.device)
            action = self.policy.run(None, {key: obs_buf_dict[key].cpu().numpy() for key in obs_buf_dict})[0]
            start=time.time()
            obs_buf_dict = self.env.step(action)
            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step>0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f'Time until next step is negative: {time_until_next_step}')
            self.time=time.time()