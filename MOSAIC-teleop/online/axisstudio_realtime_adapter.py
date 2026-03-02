from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
from collections import deque

import numpy as np
import torch

from stgat_g1_retarget.online.realtime_adapter import RealtimeAdapter, RealtimeAdapterConfig
from stgat_g1_retarget.data_utils.common import (
    quat_to_rotation_matrix_wxyz,
    quat_yaw_wxyz,
    rotate_world_vecs_to_heading,
    quat_to_wxyz_best_effort,
)


def _build_axis_transform(
    axis_order: tuple[int, int, int] | None,
    axis_signs: tuple[float, float, float] | None,
) -> np.ndarray | None:
    if axis_order is None and axis_signs is None:
        return None
    order = axis_order if axis_order is not None else (0, 1, 2)
    signs = axis_signs if axis_signs is not None else (1.0, 1.0, 1.0)
    mat = np.zeros((3, 3), dtype=np.float32)
    for new_i, old_i in enumerate(order):
        mat[new_i, old_i] = 1.0
    mat = np.diag(np.asarray(signs, dtype=np.float32)) @ mat
    return mat


@dataclass(frozen=True)
class AxisStudioAdapterConfig:
    input_fps: float
    downsample: int = 4

    # AxisStudio coordinate system / unit conversions.
    # If AxisStudio outputs centimeters, set unit_scale=0.01.
    unit_scale: float = 1.0

    # Optional axis reorder/sign flip applied in MocapOnlinePreprocessor.
    axis_order: tuple[int, int, int] | None = None
    axis_signs: tuple[float, float, float] | None = None

    # holosoma-compatible preprocessing parameters
    # - ground alignment uses toe z-min with a mat_height offset
    # - scale defaults to robot_height/default_human_height if scale is None
    robot_height: float = 1.32
    default_human_height: float = 1.82
    mat_height: float = 0.1
    scale: float | None = None

    # Should match training.
    normalize_root: bool = True

    # Feature mode: "root" | "local" | "global" (should match training)
    # - "root": all joints relative to root (pelvis)
    # - "local": each joint relative to parent (bone vectors)
    # - "global": world coordinates (no normalization)
    feature_mode: str = "root"

    # Skeleton YAML path (required for local mode)
    skeleton_yaml_path: str | None = None

    # Velocity mode: "finite_diff" | "axisstudio"
    # - "finite_diff": compute velocity from position differences (matches training)
    # - "axisstudio": use AxisStudio's get_displacement_speed() directly (potentially higher quality)
    velocity_mode: str = "finite_diff"

    # Whether to compute and return human_base_features (9D: lin_vel_heading + rotmat_cols)
    # Must match training config (data.use_human_base_features)
    # Default is False for backward compatibility with old models
    use_human_base_features: bool = False

    # Require body_part world positions from AxisStudio; if False, will fall back to local positions.
    # Using local positions can severely distort realtime inputs (not global joint positions).
    require_body_part_positions: bool = False

    # How to handle local positions when body_part positions are unavailable.
    # - "fk": reconstruct world positions via local rotations + default local offsets (recommended)
    # - "raw": use joint.get_local_position() directly (not recommended)
    local_position_mode: str = "fk"


def default_axisstudio_name_aliases() -> dict[str, str]:
    """Common fallbacks when expected skeleton has extra joints."""

    return {
        "LeftToeBase": "LeftFoot",
        "RightToeBase": "RightFoot",
        "LeftFootMod": "LeftFoot",
        "RightFootMod": "RightFoot",
    }


class AxisStudioRealtimeAdapter:
    """AxisStudio MocapApi MCPAvatar -> model x_human frame (Jh,6).

    This adapter:
    - extracts per-joint world positions from AxisStudio (MCPAvatar/MCPJoint)
    - optionally rescales units (e.g. cm->m)
    - reorders joints by name into the expected skeleton order
    - runs the same online preprocessing as the project (ground align/scale/vel/root norm)

    Notes:
    - We intentionally rely on MocapOnlinePreprocessor velocity (finite difference) to
      match training/offline semantics, even though AxisStudio can provide displacement speed.
    - If your expected skeleton contains ToeBase/FootMod but AxisStudio doesn't, we alias
      them to Foot by default (configurable).
    """

    def __init__(
        self,
        *,
        expected_joint_names: list[str],
        axisstudio_cfg: AxisStudioAdapterConfig,
        name_aliases: dict[str, str] | None = None,
    ):
        self._expected_joint_names = list(expected_joint_names)
        self._cfg = axisstudio_cfg
        self._name_aliases = dict(default_axisstudio_name_aliases())
        if name_aliases:
            self._name_aliases.update(dict(name_aliases))

        self._adapter: RealtimeAdapter | None = None
        self._incoming_joint_names: list[str] | None = None
        self._incoming_joints: list[Any] | None = None
        self._last_pos_raw_stream: np.ndarray | None = None
        self._root_joint_idx: int = 0
        self._last_root_pos_axis: np.ndarray | None = None
        self._last_root_quat_axis: np.ndarray | None = None
        self._local_parent_indices: list[int] | None = None
        self._local_fk_order: list[int] | None = None
        self._local_root_idx: int | None = None
        self._local_offsets: np.ndarray | None = None

        # 用于计算 human_base_features 的历史缓冲
        self._root_vel_history: deque = deque(maxlen=2)  # 保存根节点速度历史（用于平滑）
        self._root_quat_history: deque = deque(maxlen=2)  # 保存根节点四元数历史
        self._root_pos_history: deque = deque(maxlen=2)  # 保存根节点位置历史（用于 finite-diff 速度回退）

        # 用于 AxisStudio 速度模式的缓冲
        self._joint_vel_buffer: Optional[np.ndarray] = None  # 保存上一帧的关节速度（用于平滑）

        # Position source mode (auto-detected on first avatar init)
        # - "body_part": use joint.get_body_part().get_position() for all joints
        # - "local": use joint.get_local_position() for all joints
        # NOTE: Mixing sources per-joint is disastrous (root becomes world, others local).
        self._pos_source_mode: str | None = None
        self._axis_mat: np.ndarray | None = _build_axis_transform(
            axisstudio_cfg.axis_order, axisstudio_cfg.axis_signs
        )

    @property
    def adapter(self) -> RealtimeAdapter:
        if self._adapter is None:
            raise RuntimeError("AxisStudioRealtimeAdapter is not initialized yet; call update_from_avatar() once.")
        return self._adapter

    @property
    def incoming_joint_names(self) -> list[str] | None:
        return self._incoming_joint_names

    @property
    def pos_source_mode(self) -> str | None:
        return self._pos_source_mode

    @property
    def last_pos_raw_stream(self) -> np.ndarray | None:
        """Last raw joint positions from MocapApi (incoming order, before unit_scale)."""
        if self._last_pos_raw_stream is None:
            return None
        return self._last_pos_raw_stream.copy()

    @property
    def last_root_pos_axis(self) -> np.ndarray | None:
        """Last root position after axis alignment (meters, before scale)."""
        if self._last_root_pos_axis is None:
            return None
        return self._last_root_pos_axis.copy()

    @property
    def last_root_quat_axis(self) -> np.ndarray | None:
        """Last root orientation after axis alignment (wxyz)."""
        if self._last_root_quat_axis is None:
            return None
        return self._last_root_quat_axis.copy()

    def reset(self) -> None:
        if self._adapter is not None:
            self._adapter.reset()
        self._root_vel_history.clear()
        self._root_quat_history.clear()
        self._root_pos_history.clear()
        self._joint_vel_buffer = None
        self._last_pos_raw_stream = None
        self._last_root_pos_axis = None
        self._last_root_quat_axis = None
        self._local_parent_indices = None
        self._local_fk_order = None
        self._local_root_idx = None
        self._local_offsets = None

    def _get_joint_body_part_position(self, joint: Any) -> np.ndarray | None:
        """Return body-part world position if available, else None."""
        try:
            body = joint.get_body_part()
            x, y, z = body.get_position()
            return np.array([float(x), float(y), float(z)], dtype=np.float32)
        except Exception:
            return None

    def _get_joint_local_position(self, joint: Any) -> np.ndarray | None:
        """Return joint local position if available, else None."""
        try:
            p = joint.get_local_position()
            if p is None:
                return None
            x, y, z = p
            return np.array([float(x), float(y), float(z)], dtype=np.float32)
        except Exception:
            return None

    def _read_joint_position(self, joint: Any) -> np.ndarray:
        """Read joint position using the globally chosen source mode."""
        mode = self._pos_source_mode or "body_part"
        if mode == "body_part":
            p = self._get_joint_body_part_position(joint)
            if p is not None:
                return p
            return np.zeros(3, dtype=np.float32)
        # local
        p = self._get_joint_local_position(joint)
        if p is not None:
            return p
        return np.zeros(3, dtype=np.float32)

    @staticmethod
    def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Quaternion multiply (wxyz)."""
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _quat_conj(q: np.ndarray) -> np.ndarray:
        out = q.copy()
        out[1:] *= -1.0
        return out

    def _quat_rotate(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Rotate vector v by quaternion q (wxyz)."""
        qv = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
        return self._quat_mul(self._quat_mul(q, qv), self._quat_conj(q))[1:4]

    def _init_local_fk(self, avatar: Any, joints: list[Any]) -> None:
        """Build parent indices + offsets for local-FK fallback."""
        tags: list[int] = []
        for j in joints:
            try:
                tags.append(int(j.get_tag()))
            except Exception:
                tags.append(-1)
        tag_to_idx = {t: i for i, t in enumerate(tags) if t >= 0}

        parent_idx = [-1 for _ in joints]
        for i, j in enumerate(joints):
            try:
                tag = tags[i]
                parent_tag = int(j.get_parent_joint_tag(tag))
            except Exception:
                parent_tag = -1
            if parent_tag in tag_to_idx:
                parent_idx[i] = int(tag_to_idx[parent_tag])

        root_idx = 0
        try:
            root_tag = int(avatar.get_root_joint().get_tag())
            if root_tag in tag_to_idx:
                root_idx = int(tag_to_idx[root_tag])
        except Exception:
            root_idx = 0

        children = [[] for _ in joints]
        for i, p in enumerate(parent_idx):
            if p >= 0:
                children[p].append(i)

        order: list[int] = []
        queue = [root_idx]
        seen = {root_idx}
        while queue:
            u = queue.pop(0)
            order.append(u)
            for v in children[u]:
                if v not in seen:
                    seen.add(v)
                    queue.append(v)
        for i in range(len(joints)):
            if i not in seen:
                order.append(i)

        offsets = np.zeros((len(joints), 3), dtype=np.float32)
        for i, j in enumerate(joints):
            pos = None
            try:
                pos = j.get_default_local_position()
            except Exception:
                pos = None
            if pos is None:
                try:
                    pos = j.get_local_position()
                except Exception:
                    pos = None
            if pos is not None:
                offsets[i] = np.array([float(pos[0]), float(pos[1]), float(pos[2])], dtype=np.float32)

        self._local_parent_indices = parent_idx
        self._local_fk_order = order
        self._local_root_idx = root_idx
        self._local_offsets = offsets

    def _compute_world_pos_from_local(self, avatar: Any) -> np.ndarray:
        """Reconstruct world positions from local rotations + offsets."""
        assert self._incoming_joints is not None
        joints = self._incoming_joints
        if self._local_parent_indices is None or self._local_fk_order is None or self._local_offsets is None:
            self._init_local_fk(avatar, joints)

        J = len(joints)
        local_rot = np.zeros((J, 4), dtype=np.float32)
        for i, j in enumerate(joints):
            try:
                w, x, y, z = j.get_local_rotation()
                local_rot[i] = np.array([float(w), float(x), float(y), float(z)], dtype=np.float32)
            except Exception:
                local_rot[i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        # normalize
        n = np.linalg.norm(local_rot, axis=1, keepdims=True)
        n = np.clip(n, 1e-8, None)
        local_rot = local_rot / n

        world_rot = np.zeros_like(local_rot, dtype=np.float32)
        world_pos = np.zeros((J, 3), dtype=np.float32)
        root_idx = int(self._local_root_idx or 0)
        root_joint = joints[root_idx]
        root_pos = self._get_joint_local_position(root_joint)
        if root_pos is None:
            root_pos = np.zeros(3, dtype=np.float32)
        world_rot[root_idx] = local_rot[root_idx]
        world_pos[root_idx] = root_pos

        order = list(self._local_fk_order or [])
        for idx in order:
            if idx == root_idx:
                continue
            p = int(self._local_parent_indices[idx]) if self._local_parent_indices is not None else -1
            if p < 0:
                world_rot[idx] = local_rot[idx]
                world_pos[idx] = world_pos[root_idx] + self._local_offsets[idx]
                continue
            world_rot[idx] = self._quat_mul(world_rot[p], local_rot[idx])
            world_pos[idx] = world_pos[p] + self._quat_rotate(world_rot[p], self._local_offsets[idx])

        return world_pos

    def _maybe_promote_root_local_to_world(self, pos: np.ndarray) -> np.ndarray:
        """Fix a common AxisStudio stream quirk when using get_local_position().

        Some streams report:
        - root joint (e.g., Hips) in world coordinates
        - all other joints in a root-local character space (small values)

        If we later apply root normalization (subtract root from all joints), this becomes
        catastrophic (children already root-relative -> huge bogus offsets).

        This function detects that pattern and converts children to world coordinates by
        adding the root translation back: p_world[j] = p_local[j] + p_root_world.
        """
        if self._pos_source_mode != "local":
            return pos
        if pos.ndim != 2 or pos.shape[1] != 3 or pos.shape[0] < 2:
            return pos

        # Identify root index (prefer "Hips" if present).
        root_idx = 0
        if self._incoming_joint_names:
            for i, name in enumerate(self._incoming_joint_names):
                if name in ("Hips", "Pelvis"):
                    root_idx = i
                    break

        root = pos[root_idx]
        if not np.all(np.isfinite(root)):
            return pos

        root_norm = float(np.linalg.norm(root))
        child_norms = np.linalg.norm(pos, axis=1)
        if child_norms.shape[0] <= 1:
            return pos
        child_norms_wo_root = np.delete(child_norms, root_idx)
        median_child = float(np.median(child_norms_wo_root))

        # Heuristic thresholds in *raw* units (typically cm when unit_scale=0.01).
        # Pattern: root is far from origin, children are near origin (root-local).
        if root_norm < 100.0:
            return pos
        if median_child > 120.0:
            return pos
        if root_norm / max(median_child, 1e-6) < 5.0:
            return pos

        promoted = pos.copy()
        for j in range(promoted.shape[0]):
            if j == root_idx:
                continue
            promoted[j] = promoted[j] + root

        if self.debug and not getattr(self, "_printed_root_local_promotion", False):
            print(
                "[AxisStudio] Detected root-in-world + children-root-local positions from get_local_position(); "
                "promoting children to world by adding root translation for consistency."
            )
            self._printed_root_local_promotion = True

        return promoted

    def _safe_get_joint_world_velocity(self, joint: Any) -> np.ndarray:
        """Best-effort joint velocity.

        Preferred: body_part.get_displacement_speed().
        Fallback: zeros (velocity can be finite-diff'ed later by RealtimeAdapter).
        """
        try:
            body = joint.get_body_part()
            vx, vy, vz = body.get_displacement_speed()
            return np.array([float(vx), float(vy), float(vz)], dtype=np.float32)
        except Exception:
            return np.zeros(3, dtype=np.float32)

    def _safe_get_root_quat_wxyz(self, root_joint: Any) -> np.ndarray:
        """Best-effort root orientation as wxyz."""
        # 1) Some streams provide posture on body part
        try:
            body = root_joint.get_body_part()
            w, x, y, z = body.get_posture()
            return np.array([float(w), float(x), float(y), float(z)], dtype=np.float32)
        except Exception:
            pass

        # 2) Joint local rotation (wrapper returns wxyz)
        try:
            w, x, y, z = root_joint.get_local_rotation()
            return np.array([float(w), float(x), float(y), float(z)], dtype=np.float32)
        except Exception:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    @staticmethod
    def _rotation_matrix_to_quat_wxyz(rot: np.ndarray) -> np.ndarray:
        m = np.asarray(rot, dtype=np.float64).reshape(3, 3)
        t = float(np.trace(m))
        if t > 0.0:
            s = np.sqrt(t + 1.0) * 2.0
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        else:
            if m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
                s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
                w = (m[2, 1] - m[1, 2]) / s
                x = 0.25 * s
                y = (m[0, 1] + m[1, 0]) / s
                z = (m[0, 2] + m[2, 0]) / s
            elif m[1, 1] > m[2, 2]:
                s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
                w = (m[0, 2] - m[2, 0]) / s
                x = (m[0, 1] + m[1, 0]) / s
                y = 0.25 * s
                z = (m[1, 2] + m[2, 1]) / s
            else:
                s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
                w = (m[1, 0] - m[0, 1]) / s
                x = (m[0, 2] + m[2, 0]) / s
                y = (m[1, 2] + m[2, 1]) / s
                z = 0.25 * s
        q = np.array([w, x, y, z], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n > 1e-8:
            q = q / n
        return q

    def _apply_axis_transform_to_pose(
        self, pos: np.ndarray, quat_wxyz: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._axis_mat is None:
            return pos.astype(np.float32), quat_wxyz.astype(np.float32)
        pos_new = self._axis_mat @ pos
        rot = quat_to_rotation_matrix_wxyz(
            np.asarray(quat_wxyz, dtype=np.float32).reshape(1, 4)
        )[0]
        rot_new = self._axis_mat @ rot @ self._axis_mat.T
        quat_new = self._rotation_matrix_to_quat_wxyz(rot_new)
        return pos_new.astype(np.float32), quat_new.astype(np.float32)

    @staticmethod
    def _normalize_joint_name(name: str) -> str:
        # Python bindings sometimes expose C++ enum names like "JointTag_Hips".
        if name.startswith("JointTag_"):
            return name[len("JointTag_") :]
        return name

    def _init_from_avatar(self, avatar: Any) -> None:
        joints = list(avatar.get_joints())
        incoming_names = [self._normalize_joint_name(str(j.get_name())) for j in joints]
        root_idx = 0
        for i, name in enumerate(incoming_names):
            if name in ("Hips", "Pelvis"):
                root_idx = i
                break
        self._root_joint_idx = root_idx

        # Detect a consistent position source mode.
        # If body_part positions are missing for many joints, we force using local positions for ALL joints
        # to avoid mixing frames (world root + local limbs) which yields huge bogus bone vectors.
        n_total = max(1, len(joints))
        n_body_ok = 0
        for j in joints:
            if self._get_joint_body_part_position(j) is not None:
                n_body_ok += 1
        frac = float(n_body_ok) / float(n_total)
        if frac >= 0.9:
            self._pos_source_mode = "body_part"
        else:
            if self._cfg.require_body_part_positions:
                raise RuntimeError(
                    "[AxisStudio] body_part positions are not available for most joints "
                    f"({n_body_ok}/{n_total}, {frac*100:.1f}%). "
                    "This usually means MocapApi calc data is not enabled in AxisStudio. "
                    "Enable calc data or set --local-pos-mode fk/raw to allow local fallback."
                )
            self._pos_source_mode = "local"
            if self.debug:
                print(
                    f"[AxisStudio] body_part.get_position available for {n_body_ok}/{n_total} joints ({frac*100:.1f}%). "
                    f"Falling back to local positions (mode={self._cfg.local_position_mode})."
                )

        # Ground alignment joints (should match training semantics when possible).
        # Some skeletons use ToeBase for ground alignment. AxisStudio streams often don't
        # provide ToeBase, so we alias ToeBase -> Foot to keep positions usable.
        toe_names: tuple[str, str]
        if ("LeftToeBase" in self._expected_joint_names) and ("RightToeBase" in self._expected_joint_names):
            toe_names = ("LeftToeBase", "RightToeBase")
        else:
            toe_names = ("LeftFoot", "RightFoot")

        cfg = RealtimeAdapterConfig(
            input_fps=float(self._cfg.input_fps),
            downsample=int(self._cfg.downsample),
            axis_order=self._cfg.axis_order,
            axis_signs=self._cfg.axis_signs,
            robot_height=float(self._cfg.robot_height),
            default_human_height=float(self._cfg.default_human_height),
            mat_height=float(self._cfg.mat_height),
            scale=self._cfg.scale,
            normalize_root=bool(self._cfg.normalize_root),
            incoming_joint_names=incoming_names,
            expected_joint_names=self._expected_joint_names,
            name_aliases=self._name_aliases,
            toe_names=toe_names,
            feature_mode=str(self._cfg.feature_mode),
            skeleton_yaml_path=self._cfg.skeleton_yaml_path,
        )

        self._adapter = RealtimeAdapter(cfg)
        self._incoming_joint_names = incoming_names
        self._incoming_joints = joints

    def _compute_human_base_features(
        self,
        root_vel: np.ndarray,
        root_quat: np.ndarray,
    ) -> np.ndarray:
        """计算人体基座特征 (9维)

        Args:
            root_vel: (3,) 根节点线速度（world frame）
            root_quat: (4,) 根节点四元数 (wxyz)

        Returns:
            base_features: (9,) = [lin_vel_heading(3), rotmat_col1(3), rotmat_col2(3)]
        """
        # 1. 确保四元数为 wxyz 格式
        root_quat_wxyz = quat_to_wxyz_best_effort(root_quat.reshape(1, 4)).astype(np.float64)  # (1, 4)

        # 2. 计算yaw并转到heading frame
        yaw = quat_yaw_wxyz(root_quat_wxyz)  # (1,)
        root_vel_world = root_vel.reshape(1, 3).astype(np.float64)  # (1, 3)
        root_vel_heading = rotate_world_vecs_to_heading(root_vel_world, yaw)  # (1, 3)

        # 3. 从四元数计算旋转矩阵
        rotmat = quat_to_rotation_matrix_wxyz(root_quat_wxyz)  # (1, 3, 3)

        # 4. 提取前两列 (right, forward)
        rotmat_col1 = rotmat[0, :, 0]  # (3,) - right direction
        rotmat_col2 = rotmat[0, :, 1]  # (3,) - forward direction

        # 5. 组合特征
        base_features = np.concatenate(
            [
                root_vel_heading[0],  # (3,)
                rotmat_col1,          # (3,)
                rotmat_col2,          # (3,)
            ],
            axis=0,
        )  # (9,)

        return base_features.astype(np.float32)

    def _extract_joint_velocities(self, avatar: Any) -> np.ndarray:
        """提取所有关节的速度（AxisStudio模式）"""
        assert self._incoming_joints is not None

        vel = np.zeros((len(self._incoming_joints), 3), dtype=np.float32)
        for i, joint in enumerate(self._incoming_joints):
            vel[i] = self._safe_get_joint_world_velocity(joint)

        vel *= float(self._cfg.unit_scale)  # 应用单位缩放
        return vel

    def update_from_avatar(
        self,
        avatar: Any,
        *,
        device: torch.device | str | None = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Push one AxisStudio avatar update.

        Returns:
            Tuple of (x_human, base_features):
            - x_human: torch.Tensor (Jh,6) if a downsampled frame is produced, else None
            - base_features: torch.Tensor (9,) if x_human is not None, else None
        """

        if self._adapter is None:
            self._init_from_avatar(avatar)
        assert self._incoming_joints is not None

        # 1. 提取所有关节位置（必须使用一致的 source，避免 root/world + limbs/local 的混合）
        if self._pos_source_mode == "local" and str(self._cfg.local_position_mode).lower() == "fk":
            pos = self._compute_world_pos_from_local(avatar)
        else:
            pos = np.zeros((len(self._incoming_joints), 3), dtype=np.float32)
            for i, joint in enumerate(self._incoming_joints):
                pos[i] = self._read_joint_position(joint)
        self._last_pos_raw_stream = pos.copy()

        # Debug: print raw AxisStudio data every 100 frames
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
            self._debug_enabled = getattr(self, '_debug', False)
        self._frame_count += 1

        if self._debug_enabled and self._frame_count % 100 == 0:
            print(f"\n{'='*70}")
            print(f"[AxisStudio RAW DATA] Frame {self._frame_count}")
            print(f"{'='*70}")
            print(f"  Raw pos (BEFORE unit_scale={self._cfg.unit_scale}):")
            print(f"    Range: [{pos.min():.1f}, {pos.max():.1f}]")
            print(f"    Std: {pos.std():.1f}")
            print(f"    Mean: {pos.mean():.1f}")

            # Print key joints raw positions
            print(f"\n  Key Joints (Raw, likely in cm if unit_scale=0.01):")
            key_indices = [0, 3, 6, 1, 4]  # Typically: Hips, LeftFoot, RightFoot, LeftUpLeg, RightUpLeg
            for idx in key_indices:
                if idx < len(self._incoming_joints) and self._incoming_joint_names:
                    name = self._incoming_joint_names[idx]
                    print(f"    {idx:2d} {name:15s}: [{pos[idx,0]:7.1f}, {pos[idx,1]:7.1f}, {pos[idx,2]:7.1f}]")

            # Check if data is static (all positions very similar across joints)
            pos_range = pos.max() - pos.min()
            if pos_range < 10.0:  # If all joints within 10 units (cm), likely static
                print(f"\n  ⚠ WARNING: Position range ({pos_range:.1f}) is very small!")
                print(f"    This suggests either:")
                print(f"    1) All joints at nearly same position (bad data)")
                print(f"    2) Person in T-pose at origin")
                print(f"    3) AxisStudio not streaming live data")

            # Check variance across time (if we have history)
            if hasattr(self, '_pos_history'):
                if len(self._pos_history) >= 10:
                    pos_hist = np.array(list(self._pos_history))  # (N, J, 3)
                    temporal_std = pos_hist.std(axis=0).mean()
                    print(f"\n  Temporal variance (last 10 frames):")
                    print(f"    Mean std across joints: {temporal_std:.2f}")
                    if temporal_std < 1.0:  # Less than 1cm movement over 10 frames
                        print(f"    ⚠ Very low temporal variance - person may be stationary!")
            else:
                self._pos_history = deque(maxlen=10)
            self._pos_history.append(pos.copy())

            print(f"{'='*70}\n")

        # If we're using get_local_position(), some streams provide root in world coordinates
        # but children in root-local space. Promote to consistent world positions before
        # unit scaling + downstream preprocessing. (Skip if local-FK already produced world positions.)
        if self._pos_source_mode == "local" and str(self._cfg.local_position_mode).lower() != "fk":
            pos = self._maybe_promote_root_local_to_world(pos)

        pos *= float(self._cfg.unit_scale)

        # Cache aligned root pose (meters, before scale) for downstream consumers.
        try:
            root_idx = int(self._root_joint_idx)
            if self._incoming_joints is not None and 0 <= root_idx < len(self._incoming_joints):
                root_pos = pos[root_idx].copy()
                root_quat = self._safe_get_root_quat_wxyz(self._incoming_joints[root_idx])
                root_pos, root_quat = self._apply_axis_transform_to_pose(root_pos, root_quat)
                self._last_root_pos_axis = root_pos
                self._last_root_quat_axis = root_quat
        except Exception:
            pass

        # 2. 提取根节点（第一个关节，通常是Hips/Pelvis）的速度和四元数（用于 base_features）
        # 只有在启用 base_features 时才提取
        if self._cfg.use_human_base_features:
            try:
                root_joint = self._incoming_joints[0]  # 第一个关节为根节点

                # Root position (best-effort) for finite-diff fallback
                root_pos = self._safe_get_joint_world_position(root_joint) * float(self._cfg.unit_scale)
                self._root_pos_history.append(root_pos)

                # Root velocity: prefer AxisStudio displacement speed; fallback to finite-diff on root position
                root_vel = self._safe_get_joint_world_velocity(root_joint) * float(self._cfg.unit_scale)
                if np.allclose(root_vel, 0.0) and len(self._root_pos_history) >= 2:
                    dt = 1.0 / float(self._cfg.input_fps)
                    if dt > 0:
                        root_vel = (self._root_pos_history[-1] - self._root_pos_history[-2]) / dt

                # Apply the same axis fix + scale as the mocap preprocessor so base features are in the aligned frame.
                try:
                    if self.adapter is not None:
                        pre = self.adapter._pre
                        if pre.axis_order is not None:
                            root_vel = root_vel[list(pre.axis_order)]
                        if pre.axis_signs is not None:
                            root_vel = root_vel * np.asarray(pre.axis_signs, dtype=root_vel.dtype)
                        root_vel = root_vel * float(pre.scale)
                except Exception:
                    pass

                # Root quaternion (wxyz)
                root_quat_wxyz = self._safe_get_root_quat_wxyz(root_joint)

                # 保存到历史缓冲
                self._root_vel_history.append(root_vel)
                self._root_quat_history.append(root_quat_wxyz)

            except Exception as e:
                # 如果提取失败，使用默认值
                root_vel = np.zeros(3, dtype=np.float32)
                root_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
                self._root_vel_history.append(root_vel)
                self._root_quat_history.append(root_quat_wxyz)
                if self.debug:
                    print(f"Warning: Failed to extract root velocity/quaternion from AxisStudio: {e}")

        # 3. 根据配置选择速度源
        velocity_mode = str(self._cfg.velocity_mode).lower()

        if velocity_mode == "axisstudio":
            # 使用 AxisStudio 提供的速度（可能更准确但与训练有分布偏移）
            try:
                joint_vel = self._extract_joint_velocities(avatar)

                # 平滑处理：与上一帧速度进行加权平均
                if self._joint_vel_buffer is not None:
                    # 简单移动平均：0.7 * current + 0.3 * previous
                    joint_vel = 0.7 * joint_vel + 0.3 * self._joint_vel_buffer

                self._joint_vel_buffer = joint_vel.copy()

                # 调用底层适配器，并在之后注入速度
                x_human = self.adapter.update(pos, device=device)

                # 如果产生了输出，替换速度部分
                if x_human is not None and self.adapter._pre._vel_buf:
                    # 获取底层预处理器的重排序索引
                    if self.adapter._index_map is not None:
                        joint_vel_reordered = joint_vel[self.adapter._index_map]
                    else:
                        joint_vel_reordered = joint_vel

                    # 应用与位置相同的预处理（地面对齐、缩放、归一化）
                    # 注意：速度不需要地面对齐，但需要缩放和归一化
                    vel_processed = joint_vel_reordered
                    try:
                        if self.adapter._pre.axis_order is not None:
                            vel_processed = vel_processed[:, list(self.adapter._pre.axis_order)]
                        if self.adapter._pre.axis_signs is not None:
                            vel_processed = vel_processed * np.asarray(self.adapter._pre.axis_signs, dtype=vel_processed.dtype)
                    except Exception:
                        pass
                    vel_processed = vel_processed * float(self.adapter._pre.scale)

                    if self.adapter._pre.normalize_root:
                        root_v = vel_processed[self.adapter._pre.root_idx]
                        vel_processed = vel_processed - root_v

                    # 替换底层缓冲区中的速度
                    if len(self.adapter._pre._vel_buf) > 0:
                        self.adapter._pre._vel_buf[-1] = vel_processed

                        # 重新构建 x_human（位置+新速度）
                        if len(self.adapter._pre._pos_buf) > 0:
                            pos_last = self.adapter._pre._pos_buf[-1]
                            x_last = np.concatenate([pos_last, vel_processed], axis=-1)  # (J, 6)
                            x_human = torch.from_numpy(x_last.astype(np.float32))
                            if device is not None:
                                x_human = x_human.to(device)

            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to use AxisStudio velocities, falling back to finite_diff: {e}")
                # 回退到差分计算
                x_human = self.adapter.update(pos, device=device)
        else:
            # 默认：使用有限差分计算速度（与训练一致）
            x_human = self.adapter.update(pos, device=device)

        # 4. 如果产生了下采样帧，计算 base_features（仅在启用时）
        if x_human is not None:
            # 只有在启用 use_human_base_features 时才计算 base_features
            if self._cfg.use_human_base_features:
                # 使用最新的速度和四元数计算 base_features
                if len(self._root_vel_history) > 0 and len(self._root_quat_history) > 0:
                    try:
                        base_feat_np = self._compute_human_base_features(
                            self._root_vel_history[-1],
                            self._root_quat_history[-1],
                        )
                        # 转换为 torch tensor
                        if device is None:
                            device = x_human.device
                        base_features = torch.from_numpy(base_feat_np).to(device=device, dtype=x_human.dtype)
                    except Exception as e:
                        if self.debug:
                            print(f"Warning: Failed to compute base_features: {e}")
                        base_features = None
                else:
                    base_features = None
            else:
                # 向后兼容：不使用 base_features 的旧模型
                base_features = None

            return x_human, base_features
        else:
            return None, None

    @property
    def debug(self) -> bool:
        """是否启用调试输出"""
        return getattr(self, '_debug', False)

    def set_debug(self, enabled: bool) -> None:
        """设置调试模式"""
        self._debug = bool(enabled)
        self._debug_enabled = bool(enabled)
