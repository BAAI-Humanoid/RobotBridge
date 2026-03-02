from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch

from stgat_g1_retarget.data_utils.mocap.online_preprocess import MocapOnlinePreprocessor


def build_name_index_map(
    *,
    incoming_names: Iterable[str],
    expected_names: Iterable[str],
    name_aliases: dict[str, str] | None = None,
) -> np.ndarray:
    """Build an index map so that pos_expected = pos_incoming[index_map].

    Args:
        incoming_names: Joint names provided by the upstream system.
        expected_names: Joint names order expected by the network/preprocessor.

    Returns:
        index_map: int array of shape (J_expected,)

    Raises:
        KeyError if any expected joint is missing from incoming.
    """

    incoming = list(incoming_names)
    expected = list(expected_names)
    lut = {n: i for i, n in enumerate(incoming)}
    aliases = dict(name_aliases or {})

    missing: list[str] = []
    resolved: list[int] = []
    for exp_name in expected:
        if exp_name in lut:
            resolved.append(lut[exp_name])
            continue
        alias = aliases.get(exp_name)
        if alias is not None and alias in lut:
            resolved.append(lut[alias])
            continue
        missing.append(exp_name)
    if missing:
        raise KeyError(f"Incoming joints missing {len(missing)} expected names (first 10): {missing[:10]}")
    return np.asarray(resolved, dtype=np.int64)


@dataclass(frozen=True)
class RealtimeAdapterConfig:
    # Raw stream rate (before downsample)
    input_fps: float

    # Upstream joint ordering; if provided and differs from expected_joint_names,
    # we will reorder using joint names.
    incoming_joint_names: list[str] | None = None

    # Expected order for the model. If None, MocapOnlinePreprocessor falls back to
    # configs/humans/mocap.yaml.
    expected_joint_names: list[str] | None = None

    # Optional mapping from expected joint name -> incoming joint name.
    name_aliases: dict[str, str] | None = None

    # Mirror holosoma preprocessing defaults
    downsample: int = 4

    # How many processed frames to keep in the underlying preprocessor buffer.
    # For realtime inference we typically only need the latest frame, so default to 1
    # to avoid unbounded memory growth.
    window_len: int | None = 1
    robot_height: float = 1.32
    default_human_height: float = 1.78
    mat_height: float = 0.1

    # Optional explicit scale override. If set, MocapOnlinePreprocessor uses this value
    # instead of robot_height/default_human_height.
    scale: float | None = None

    # Coordinate system fixes (optional)
    axis_order: tuple[int, int, int] | None = None
    axis_signs: tuple[float, float, float] | None = None

    # Root-relative normalization (should match training)
    normalize_root: bool = True
    root_name: str = "Hips"

    # Feature mode: "root" | "local" | "global" (should match training)
    # - "root": all joints relative to root (pelvis)
    # - "local": each joint relative to parent (bone vectors)
    # - "global": world coordinates (no normalization)
    feature_mode: str = "root"

    # Skeleton YAML path (required for local mode)
    skeleton_yaml_path: str | None = None

    # For ground alignment. If your skeleton has no toes, set to feet.
    toe_names: tuple[str, str] = ("LeftToeBase", "RightToeBase")


class RealtimeAdapter:
    """Convert live joint positions (J,3) frames to the model's human input (Jh,6).

    Pipeline (holosoma-compatible):
    - optional axis reorder/sign flip
    - downsample by fixed factor
    - ground alignment via toe/foot z-min (with mat_height)
    - scale to robot size
    - finite-difference velocity
    - optional root-relative normalization

    Note:
        If your upstream joint ordering differs from expected_joint_names,
        pass incoming_joint_names so we can reorder by name.
    """

    def __init__(self, cfg: RealtimeAdapterConfig):
        self.cfg = cfg

        self._pre = MocapOnlinePreprocessor(
            input_fps=float(cfg.input_fps),
            joint_names=cfg.expected_joint_names,
            window_len=int(cfg.window_len) if cfg.window_len is not None else None,
            normalize_root=bool(cfg.normalize_root),
            feature_mode=str(cfg.feature_mode or "root"),
            skeleton_yaml_path=cfg.skeleton_yaml_path,
            root_name=str(cfg.root_name),
            toe_names=tuple(cfg.toe_names),
            downsample=int(cfg.downsample),
            robot_height=float(cfg.robot_height),
            default_human_height=float(cfg.default_human_height),
            scale=cfg.scale,
            mat_height=float(cfg.mat_height),
            axis_order=cfg.axis_order,
            axis_signs=cfg.axis_signs,
        )

        # Debug/recording hooks (kept lightweight; storing only the most recent processed frame).
        self._last_pos_expected_stream_m: np.ndarray | None = None  # (J,3) after name reorder, before axis/scale/ground
        self._last_preprocess: dict[str, np.ndarray] | None = None

        self._index_map: np.ndarray | None = None
        if cfg.incoming_joint_names is not None:
            incoming_names = list(cfg.incoming_joint_names)
            expected_for_map = list(self._pre.joint_names)
            if incoming_names != expected_for_map:
                self._index_map = build_name_index_map(
                    incoming_names=incoming_names,
                    expected_names=expected_for_map,
                    name_aliases=cfg.name_aliases,
                )

    @property
    def fps(self) -> float:
        """Effective fps after downsample."""

        return float(self._pre.fps)

    @property
    def calibrated(self) -> bool:
        return bool(self._pre.calibrated)

    def reset(self) -> None:
        self._pre.reset()
        self._last_pos_expected_stream_m = None
        self._last_preprocess = None

    @property
    def last_pos_expected_stream_m(self) -> np.ndarray | None:
        """Last upstream frame in expected joint order (meters), before axis/scale/ground align."""
        return None if self._last_pos_expected_stream_m is None else self._last_pos_expected_stream_m.copy()

    @property
    def last_human_pos(self) -> np.ndarray | None:
        """Last preprocessed human position frame (J,3) after axis/ground/scale/(root/local)."""
        if self._last_preprocess is None:
            return None
        hp = self._last_preprocess.get("human_pos")
        if hp is None or hp.ndim != 3 or hp.shape[0] < 1:
            return None
        return np.asarray(hp[-1], dtype=np.float32)

    @property
    def last_x_human(self) -> np.ndarray | None:
        """Last model input frame (J,6) as numpy, if available."""
        if self._last_preprocess is None:
            return None
        x = self._last_preprocess.get("x_human")
        if x is None or x.ndim != 4:
            return None
        return np.asarray(x[0, -1], dtype=np.float32)

    def update(self, pos_frame: np.ndarray, *, device: torch.device | str | None = None) -> torch.Tensor | None:
        """Push one upstream frame and return one model input frame (Jh,6).

        Returns None when the frame is skipped by downsample.
        """

        pos = np.asarray(pos_frame, dtype=np.float32)
        if self._index_map is not None:
            pos = pos[self._index_map]

        # Keep a copy for recording/debugging (expected joint order, meters).
        self._last_pos_expected_stream_m = pos.copy()

        res = self._pre.update(pos)
        if res is None:
            return None

        # Cache last preprocess result (typically contains a single frame when window_len=1).
        self._last_preprocess = res

        x_last = np.asarray(res["x_human"][0, -1], dtype=np.float32)
        out = torch.from_numpy(x_last)
        if device is not None:
            out = out.to(device)
        return out
