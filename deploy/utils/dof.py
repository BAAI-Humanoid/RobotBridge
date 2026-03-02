from typing import List, Optional
import numpy as np


class DoFAdapter:
    def __init__(self, src_joint_names: List[str], tar_joint_names: List[str]):
        self.src_joint_names = list(src_joint_names)
        self.tar_joint_names = list(tar_joint_names)
        self.src_len = len(self.src_joint_names)
        self.tar_len = len(self.tar_joint_names)

        self.src_indices: List[int] = []
        self.tar_indices: List[int] = []
        for i, name in enumerate(self.src_joint_names):
            if name in self.tar_joint_names:
                self.src_indices.append(i)
                self.tar_indices.append(self.tar_joint_names.index(name))

        if not self.src_indices:
            raise ValueError("Failed to build DoF adapter mapping; joint names do not overlap.")

    def fit(self, data: np.ndarray, template: Optional[np.ndarray] = None) -> np.ndarray:
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError("Only 1D arrays are supported by _DoFAdapter.")
        if arr.shape[0] != self.src_len:
            raise ValueError(f"Input length {arr.shape[0]} does not match source joints {self.src_len}.")

        if template is None:
            result = np.zeros(self.tar_len, dtype=np.float32)
        else:
            result = np.asarray(template, dtype=np.float32).copy()
            if result.shape[0] != self.tar_len:
                raise ValueError("Template length does not match target joints.")

        result[self.tar_indices] = arr[self.src_indices]
        return result