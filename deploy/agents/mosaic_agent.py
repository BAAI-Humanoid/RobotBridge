import time

import numpy as np
from loguru import logger

from agents.base_agent import BaseAgent
from utils.dataset import MosaicModelMeta


class MosaicAgent(BaseAgent):
    """Agent wrapper that loads Mosaic ONNX metadata and wires it to the environment."""

    def load_onnx_policy(self):
        super().load_onnx_policy()
        model_meta = MosaicModelMeta.from_onnx_session(self.policy)
        if not hasattr(self.env, "configure_from_modelmeta"):
            raise AttributeError("Environment does not support Mosaic metadata configuration.")
        self.env.configure_from_modelmeta(model_meta)
        logger.info(
            "Mosaic policy metadata loaded. Joints: {} Anchor body: {}",
            len(model_meta.joint_names),
            model_meta.anchor_body_name,
        )

    def run(self):
        obs_buf_dict = self.env.reset()
        self.time = time.time()
        while True:
            inputs = {key: obs_buf_dict[key].astype(np.float32) for key in obs_buf_dict}
            ort_outputs = self.policy.run(None, inputs)
            action = ort_outputs[0]

            obs_buf_dict = self.env.step(action)
            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Time until next step is negative: {time_until_next_step}")
            self.time = time.time()

            if self.env.motion_loader.cur_motion_end:
                obs_buf_dict = self.env.next_motion()
            
    def run_eval(self):
        obs_buf_dict = self.env.reset()
        self.time = time.time()
        while True:
            inputs = {key: obs_buf_dict[key].astype(np.float32) for key in obs_buf_dict}
            ort_outputs = self.policy.run(None, inputs)
            action = ort_outputs[0]

            obs_buf_dict = self.env.step(action)
            if self.env.simulator.is_real:
                time_until_next_step = self.env.simulator.high_dt - (time.time() - self.time)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
                else:
                    logger.warning(f"Time until next step is negative: {time_until_next_step}")
            self.time = time.time()    

            if self.env.motion_loader.cur_motion_end:
                obs_buf_dict = self.env.next_motion()

