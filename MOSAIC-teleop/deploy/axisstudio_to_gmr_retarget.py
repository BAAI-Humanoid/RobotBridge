#!/usr/bin/env python3
"""AxisStudio MocapApi -> GMR realtime retargeting.

This script mirrors AxisStudio MocapApi streaming (like inference_axisstudio_realtime.py),
but uses GMR (general_motion_retargeting) for retargeting instead of STGAT.
"""

from __future__ import annotations

import sys
import time
import argparse
from pathlib import Path
from typing import Optional
from datetime import datetime
import threading

import numpy as np
from loop_rate_limiters import RateLimiter

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# AxisStudio MocapApi
try:
    from online.mocap_robotapi import (
        MCPApplication,
        MCPEventType,
        MCPSettings,
        MCPAvatar,
    )
    MOCAP_API_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"Warning: AxisStudio MocapApi not available: {e}")
    MOCAP_API_AVAILABLE = False


def _normalize_joint_name(name: str) -> str:
    if name.startswith("JointTag_"):
        return name[len("JointTag_") :]
    return name


def _default_name_aliases() -> dict[str, str]:
    return {
        "LeftToeBase": "LeftFoot",
        "RightToeBase": "RightFoot",
        "LeftFootMod": "LeftFoot",
        "RightFootMod": "RightFoot",
    }


def _build_axis_transform(axis_order: tuple[int, int, int] | None,
                          axis_signs: tuple[float, float, float] | None) -> Optional[np.ndarray]:
    if axis_order is None and axis_signs is None:
        return None
    order = axis_order if axis_order is not None else (0, 1, 2)
    signs = axis_signs if axis_signs is not None else (1.0, 1.0, 1.0)
    mat = np.zeros((3, 3), dtype=np.float32)
    for new_i, old_i in enumerate(order):
        mat[new_i, old_i] = 1.0
    mat = np.diag(np.asarray(signs, dtype=np.float32)) @ mat
    return mat


def _apply_axis_transform_to_pos_quat(pos: np.ndarray, quat_wxyz: np.ndarray, mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # pos: (3,), quat_wxyz: (4,), mat: (3,3)
    pos_new = mat @ pos
    try:
        from scipy.spatial.transform import Rotation as R
        rot = R.from_quat(quat_wxyz, scalar_first=True)
        rmat = rot.as_matrix()
        rmat_new = mat @ rmat @ mat.T
        quat_new = R.from_matrix(rmat_new).as_quat(scalar_first=True)
        return pos_new.astype(np.float32), quat_new.astype(np.float32)
    except Exception:
        return pos_new.astype(np.float32), quat_wxyz.astype(np.float32)


def _quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
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


def _quat_conj(q: np.ndarray) -> np.ndarray:
    out = q.copy()
    out[1:] *= -1.0
    return out


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, float(v[0]), float(v[1]), float(v[2])], dtype=np.float32)
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:4]


_G1_LCM_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


class AxisStudioGMRAdapter:
    def __init__(
        self,
        *,
        expected_human_names: list[str],
        name_aliases: dict[str, str] | None = None,
        unit_scale: float = 1.0,
        axis_order: tuple[int, int, int] | None = None,
        axis_signs: tuple[float, float, float] | None = None,
        local_pos_mode: str = "fk",
        strict_body_part: bool = False,
        debug: bool = False,
    ) -> None:
        self.expected_human_names = list(expected_human_names)
        self.name_aliases = dict(_default_name_aliases())
        if name_aliases:
            self.name_aliases.update(dict(name_aliases))
        self.unit_scale = float(unit_scale)
        self.axis_mat = _build_axis_transform(axis_order, axis_signs)
        self.local_pos_mode = str(local_pos_mode or "fk").lower()
        self.strict_body_part = bool(strict_body_part)
        self.debug = bool(debug)

        self._incoming_joints = None
        self._incoming_names = None
        self._index_map = None
        self._pos_source_mode = None

        # Local-FK state
        self._local_parent_indices = None
        self._local_fk_order = None
        self._local_root_idx = None
        self._local_offsets = None

    def _init_local_fk(self, avatar, joints) -> None:
        tags = []
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

        order = []
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

    def _compute_world_from_local(self, avatar) -> tuple[np.ndarray, np.ndarray]:
        assert self._incoming_joints is not None
        joints = self._incoming_joints
        if self._local_parent_indices is None:
            self._init_local_fk(avatar, joints)

        J = len(joints)
        local_rot = np.zeros((J, 4), dtype=np.float32)
        for i, j in enumerate(joints):
            try:
                w, x, y, z = j.get_local_rotation()
                local_rot[i] = np.array([float(w), float(x), float(y), float(z)], dtype=np.float32)
            except Exception:
                local_rot[i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        n = np.linalg.norm(local_rot, axis=1, keepdims=True)
        n = np.clip(n, 1e-8, None)
        local_rot = local_rot / n

        world_rot = np.zeros_like(local_rot, dtype=np.float32)
        world_pos = np.zeros((J, 3), dtype=np.float32)
        root_idx = int(self._local_root_idx or 0)
        root_joint = joints[root_idx]
        root_pos = None
        try:
            root_pos = root_joint.get_local_position()
        except Exception:
            root_pos = None
        if root_pos is None:
            root_pos = np.zeros(3, dtype=np.float32)
        else:
            root_pos = np.array([float(root_pos[0]), float(root_pos[1]), float(root_pos[2])], dtype=np.float32)

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
            world_rot[idx] = _quat_mul(world_rot[p], local_rot[idx])
            world_pos[idx] = world_pos[p] + _quat_rotate(world_rot[p], self._local_offsets[idx])

        return world_pos, world_rot

    def _init_from_avatar(self, avatar) -> None:
        joints = list(avatar.get_joints())
        incoming_names = [_normalize_joint_name(str(j.get_name())) for j in joints]

        name_to_idx = {n: i for i, n in enumerate(incoming_names)}
        index_map = {}
        missing = []
        for exp in self.expected_human_names:
            if exp in name_to_idx:
                index_map[exp] = name_to_idx[exp]
                continue
            alias = self.name_aliases.get(exp)
            if alias is not None and alias in name_to_idx:
                index_map[exp] = name_to_idx[alias]
                continue
            missing.append(exp)
        if missing:
            raise RuntimeError(f"Missing {len(missing)} required human joints: {missing[:10]}")

        n_total = max(1, len(joints))
        n_body_ok = 0
        for j in joints:
            try:
                body = j.get_body_part()
                _ = body.get_position()
                n_body_ok += 1
            except Exception:
                pass
        frac = float(n_body_ok) / float(n_total)
        if frac >= 0.9:
            self._pos_source_mode = "body_part"
        else:
            if self.strict_body_part:
                raise RuntimeError(
                    f"[AxisStudio] body_part positions unavailable ({n_body_ok}/{n_total}). "
                    "Enable calc_data or disable strict mode."
                )
            self._pos_source_mode = "local"
            if self.debug:
                print(f"[AxisStudio] body_part unavailable ({n_body_ok}/{n_total}); using local mode={self.local_pos_mode}.")

        self._incoming_joints = joints
        self._incoming_names = incoming_names
        self._index_map = index_map

    def build_human_data(self, avatar) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if self._incoming_joints is None:
            self._init_from_avatar(avatar)

        joints = self._incoming_joints
        assert joints is not None

        if self._pos_source_mode == "body_part":
            pos_all = np.zeros((len(joints), 3), dtype=np.float32)
            rot_all = np.zeros((len(joints), 4), dtype=np.float32)
            for i, j in enumerate(joints):
                try:
                    body = j.get_body_part()
                    x, y, z = body.get_position()
                    pos_all[i] = np.array([float(x), float(y), float(z)], dtype=np.float32)
                except Exception:
                    pos_all[i] = 0.0
                try:
                    w, x, y, z = j.get_body_part().get_posture()
                    rot_all[i] = np.array([float(w), float(x), float(y), float(z)], dtype=np.float32)
                except Exception:
                    rot_all[i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        else:
            if self.local_pos_mode == "fk":
                pos_all, rot_all = self._compute_world_from_local(avatar)
            else:
                pos_all = np.zeros((len(joints), 3), dtype=np.float32)
                rot_all = np.zeros((len(joints), 4), dtype=np.float32)
                for i, j in enumerate(joints):
                    try:
                        p = j.get_local_position()
                        if p is not None:
                            pos_all[i] = np.array([float(p[0]), float(p[1]), float(p[2])], dtype=np.float32)
                    except Exception:
                        pass
                    try:
                        w, x, y, z = j.get_local_rotation()
                        rot_all[i] = np.array([float(w), float(x), float(y), float(z)], dtype=np.float32)
                    except Exception:
                        rot_all[i] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        pos_all = pos_all * float(self.unit_scale)

        human_data = {}
        for exp in self.expected_human_names:
            idx = self._index_map[exp]
            pos = pos_all[idx]
            quat = rot_all[idx]
            if self.axis_mat is not None:
                pos, quat = _apply_axis_transform_to_pos_quat(pos, quat, self.axis_mat)
            human_data[exp] = (pos, quat)

        return human_data


def _collect_required_human_names(retarget) -> list[str]:
    names = []
    for _, entry in retarget.ik_match_table1.items():
        if entry and entry[0] not in names:
            names.append(entry[0])
    for _, entry in retarget.ik_match_table2.items():
        if entry and entry[0] not in names:
            names.append(entry[0])
    if retarget.human_root_name not in names:
        names.append(retarget.human_root_name)
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description="AxisStudio -> GMR realtime retargeting")
    ap.add_argument("--udp-port", type=int, default=7012)
    ap.add_argument("--unit-scale", type=float, default=0.01,
                    help="Scale applied to incoming positions (e.g. 0.01 for cm->m)")
    ap.add_argument("--no-holosoma-align", action="store_true",
                    help="Disable default holosoma alignment (yup->zup + y flip)")
    ap.add_argument("--axis-order", type=int, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="Optional axis reorder. Example: --axis-order 0 2 1")
    ap.add_argument("--axis-signs", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"),
                    help="Optional axis sign flips. Example: --axis-signs 1 -1 1")
    ap.add_argument("--local-pos-mode", type=str, default="fk", choices=["fk", "raw"],
                    help="Fallback when body_part positions missing: fk (recommended) or raw local positions")
    ap.add_argument("--strict-body-part", action="store_true",
                    help="Abort if body_part positions are unavailable")
    ap.add_argument("--robot", type=str, default="unitree_g1")
    ap.add_argument("--human-height", type=float, default=1.85,
                    help="Actual human height for GMR scaling")
    ap.add_argument("--offset-to-ground", action="store_true",
                    help="Offset human data to ground (GMR option)")
    ap.add_argument("--visualize-human", action="store_true",
                    help="Render human frames for debugging in GMR viewer")
    ap.add_argument("--no-viewer", action="store_true",
                    help="Disable GMR viewer (useful when streaming to robot)")
    ap.add_argument("--target-fps", type=float, default=50.0,
                    help="Target loop frequency (Hz)")
    ap.add_argument("--lcm-channel", type=str, default="camera_reference_data",
                    help="LCM channel name for robot control")
    ap.add_argument("--record-dir", type=str, default=None,
                    help="Directory to save recorded robot motions (.npz).")
    ap.add_argument("--record-prefix", type=str, default="motion",
                    help="Filename prefix for recorded motions.")
    ap.add_argument("--record-auto-start", action="store_true",
                    help="Start recording immediately (otherwise press 's' to toggle).")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not MOCAP_API_AVAILABLE:
        raise RuntimeError("AxisStudio MocapApi not available. Please install the MocapApi SDK.")

    from general_motion_retargeting import GeneralMotionRetargeting as GMR
    from general_motion_retargeting import RobotMotionViewer
    from online.lcm_publisher import LCMPublisher

    if not args.no_holosoma_align:
        if args.axis_order is None:
            args.axis_order = (0, 2, 1)
        if args.axis_signs is None:
            args.axis_signs = (1.0, -1.0, 1.0)

    target_fps = float(args.target_fps) if args.target_fps else 0.0

    print("=" * 70)
    print("AxisStudio -> GMR realtime retargeting")
    print("=" * 70)
    print(f"  - UDP port: {args.udp_port}")
    print(f"  - unit_scale: {args.unit_scale}")
    print(f"  - axis_order: {args.axis_order}")
    print(f"  - axis_signs: {args.axis_signs}")
    print(f"  - local_pos_mode: {args.local_pos_mode}")
    print(f"  - strict_body_part: {args.strict_body_part}")
    print(f"  - robot: {args.robot}")
    print(f"  - src_human: axisstudio")
    print(f"  - human_height: {args.human_height}")
    print(f"  - target_fps: {target_fps}")

    retarget = GMR(
        src_human="axisstudio",
        tgt_robot=args.robot,
        actual_human_height=args.human_height,
    )
    viewer = None if args.no_viewer else RobotMotionViewer(robot_type=args.robot)
    lcm_pub = LCMPublisher(channel_name=args.lcm_channel)

    # Build joint name -> qpos index map for the GMR robot model.
    joint_name_to_qpos = {}
    for jid in range(int(retarget.model.njnt)):
        jname = retarget.model.joint(jid).name
        adr = int(retarget.model.jnt_qposadr[jid])
        joint_name_to_qpos[jname] = adr

    qpos_indices = []
    missing = []
    for name in _G1_LCM_JOINT_ORDER:
        if name in joint_name_to_qpos:
            qpos_indices.append(joint_name_to_qpos[name])
            continue
        base = name[:-len("_joint")] if name.endswith("_joint") else name
        if base in joint_name_to_qpos:
            qpos_indices.append(joint_name_to_qpos[base])
            continue
        missing.append(name)
        qpos_indices.append(None)
    if missing:
        print(f"[GMR] Warning: {len(missing)} joints not found in model: {missing[:5]}")

    required_names = _collect_required_human_names(retarget)
    adapter = AxisStudioGMRAdapter(
        expected_human_names=required_names,
        unit_scale=float(args.unit_scale),
        axis_order=tuple(args.axis_order) if args.axis_order is not None else None,
        axis_signs=tuple(args.axis_signs) if args.axis_signs is not None else None,
        local_pos_mode=args.local_pos_mode,
        strict_body_part=args.strict_body_part,
        debug=args.debug,
    )

    rate = RateLimiter(frequency=target_fps, warn=False) if target_fps > 0 else None

    app = MCPApplication()
    settings = MCPSettings()
    settings.set_udp(int(args.udp_port))
    settings.set_bvh_rotation(0)
    app.set_settings(settings)
    ok, msg = app.open()
    if not ok:
        raise RuntimeError(f"无法打开 MocapApi: {msg}")
    print(f"✓ MocapApi 已连接 (UDP:{args.udp_port})")

    record_dir = Path(args.record_dir) if args.record_dir else None
    record_enabled = record_dir is not None
    record_running = bool(record_enabled and args.record_auto_start)
    record_toggle = False
    record_failed = False
    record_saved = False
    record_buffer: list[list[float]] = []
    record_thread = None
    stop_listen = None

    if record_enabled:
        record_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ Recording enabled: dir={record_dir} prefix={args.record_prefix}")
        try:
            from sshkeyboard import listen_keyboard, stop_listening
            stop_listen = stop_listening

            def on_press(key):
                nonlocal record_toggle, record_failed
                if key == "s":
                    record_toggle = True
                elif key == "f":
                    record_failed = True
                    record_toggle = True

            record_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={"on_press": on_press, "until": None, "sequential": False},
                daemon=True,
            )
            record_thread.start()
            if not record_running:
                print("  - press 's' to start/stop recording, 'f' to discard")
        except Exception as e:
            print(f"Warning: sshkeyboard unavailable ({e}); recording uses auto-start only.")
            record_running = bool(record_enabled)

    def _save_record() -> None:
        nonlocal record_saved
        if not record_enabled or record_saved:
            return
        if not record_buffer:
            print("[record] No frames captured; skip saving.")
            record_saved = True
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = record_dir / f"{args.record_prefix}_{ts}.npz"
        J = len(_G1_LCM_JOINT_ORDER)
        dof_pos = np.asarray([row[:J] for row in record_buffer], dtype=np.float32)
        root_pos = np.asarray([row[J:J+3] for row in record_buffer], dtype=np.float32)
        root_rot = np.asarray([row[J+3:J+7] for row in record_buffer], dtype=np.float32)
        timestamp_us = np.asarray([row[J+7] for row in record_buffer], dtype=np.int64)
        fps = float(target_fps) if target_fps > 0 else 0.0
        if fps <= 0.0:
            try:
                fps = float(retarget.model.opt.timestep)
                fps = 1.0 / fps if fps > 0 else 0.0
            except Exception:
                fps = 0.0
        np.savez_compressed(
            out_path,
            dof_pos=dof_pos,
            root_pos=root_pos,
            root_rot=root_rot,
            timestamp_us=timestamp_us,
            joint_names=np.asarray(_G1_LCM_JOINT_ORDER, dtype=object),
            fps=np.float32(fps),
        )
        print(f"[record] Saved {len(record_buffer)} frames -> {out_path}")
        record_saved = True

    try:
        while True:
            events = app.poll_next_event()
            if not events:
                time.sleep(0.001)
            if events:
                for evt in events:
                    if evt.event_type != MCPEventType.AvatarUpdated:
                        continue
                    avatar = MCPAvatar(evt.event_data.avatar_handle)
                    human_data = adapter.build_human_data(avatar)
                    qpos = retarget.retarget(human_data, offset_to_ground=args.offset_to_ground)
                    qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)
                    if qpos.shape[0] < 7:
                        continue
                    root_pos = qpos[:3].tolist()
                    root_rot = qpos[3:7].tolist()
                    dof_pos = []
                    for idx in qpos_indices:
                        if idx is None or idx >= qpos.shape[0]:
                            dof_pos.append(0.0)
                        else:
                            dof_pos.append(float(qpos[int(idx)]))

                    timestamp_us = int(time.time() * 1e6)
                    lcm_pub.publish_q(dof_pos, root_pos, root_rot, timestamp_us)
                    if record_enabled and record_running:
                        record_buffer.append(
                            list(dof_pos) + list(root_pos) + list(root_rot) + [int(timestamp_us)]
                        )

                    if viewer is not None:
                        viewer.step(
                            root_pos=qpos[:3],
                            root_rot=qpos[3:7],
                            dof_pos=qpos[7:],
                            human_motion_data=human_data if args.visualize_human else None,
                            rate_limit=False,
                        )
            if record_enabled:
                if record_toggle and record_running:
                    if record_failed:
                        print("[record] Discarded (marked failed).")
                        record_buffer = []
                        record_failed = False
                    else:
                        print("[record] Save motion.")
                        _save_record()
                    record_toggle = False
                    record_running = False
                elif record_toggle and not record_running:
                    print("[record] Start recording.")
                    record_buffer = []
                    record_toggle = False
                    record_running = True
                    record_saved = False
            if rate is not None:
                rate.sleep()
    except KeyboardInterrupt:
        print("\n停止 retargeting")
    finally:
        if record_enabled and record_running and not record_failed:
            _save_record()
        if stop_listen is not None:
            try:
                stop_listen()
            except Exception:
                pass
        try:
            app.close()
        except Exception:
            pass
        try:
            if viewer is not None:
                viewer.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
