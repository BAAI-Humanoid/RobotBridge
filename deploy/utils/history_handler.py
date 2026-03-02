import numpy as np
from loguru import logger

class HistoryHandler:
    def __init__(self, obs_aux_cfg, obs_dims):
        self.obs_dims = {}
        for dic in obs_dims:
            self.obs_dims.update(dic)
        self.history = {}
        
        self.buffer_cfg={}
        for aux_key, aux_config in obs_aux_cfg.items():
            for obs_key, obs_num in aux_config.items():
                if obs_key in self.buffer_cfg:
                    self.buffer_cfg[obs_key] = max(self.buffer_cfg[obs_key], obs_num)
                else:
                    self.buffer_cfg[obs_key] = obs_num

        logger.info(f'History Handler initialized!')
        for key in self.buffer_cfg.keys():
            self.history[key] = np.zeros((self.buffer_cfg[key], self.obs_dims[key]), dtype=np.float32)
            logger.info(f'Key: {key}, Window Size: {self.buffer_cfg[key]}, Dim: {self.obs_dims[key]}')
    
    def reset(self):
        for key in self.history.keys():
            self.history[key]*=0

    def add(self, key: str, value: np.ndarray):
        assert key in self.history.keys(), f'{key} not found in history'
        val = self.history[key].copy()
        self.history[key][1:] = val[:-1]
        self.history[key][0] = value.copy()

    def query(self, key: str) -> np.ndarray:
        assert key in self.history.keys(), f'{key} not found in history'
        return self.history[key].copy() # (hist_len, obs_dim)