import collections
import numpy as np
from loguru import logger

from envs.locomotion import Locomotion
from utils.teleop import PygameKeyboardTeleop, RealStickTeleop, TerminalKeyboardTeleop
from utils.transformation import quat_rotate_inverse


class LevelLocomotion(Locomotion):
    """
    Level Locomotion environment.
    """

    def __init__(self, config):
        super().__init__(config)
        self.locomotion_cfg = getattr(self.cfg, "locomotion", {})
        self.teleop_cfg = getattr(self.cfg, "teleop", {})
        self.loco_policy_control = getattr(self.cfg, "policy_control", {})

        # Basic scaling/parameters
        self.cmd_scale = np.asarray(self.locomotion_cfg.get("cmd_scale", [2.0, 2.0, 0.25]), dtype=np.float32)
        self.height_cmd = float(self.locomotion_cfg.get("height_cmd", 0.75))
        self.ang_vel_scale = float(self.locomotion_cfg.get("ang_vel_scale", 0.25))
        self.dof_pos_scale = float(self.locomotion_cfg.get("dof_pos_scale", 1.0))
        self.dof_vel_scale = float(self.locomotion_cfg.get("dof_vel_scale", 0.05))
        self.gait_phase_init = np.asarray(self.locomotion_cfg.get("gait_phase_init", [0.38, 0.38]), dtype=np.float32)
        self.gait_period = float(self.locomotion_cfg.get("gait_period", 0.8))
        self.stand_threshold = float(self.locomotion_cfg.get("stand_threshold", 0.1))
        self.obs_history_len = int(self.locomotion_cfg.get("obs_history_len", 10))
        self.num_obs_single_cfg = int(self.locomotion_cfg.get("num_obs_single", 48))

        # Runtime state
        self.gait_phase = self.gait_phase_init.copy()
        self.flag_stand = True
        self.obs_history = collections.deque(maxlen=self.obs_history_len)
        self.single_obs_dim = None
        self.num_actions = self.simulator.num_action

        # Pre-fill zero history, keep consistent with LEVEL code
        self._reset_history()
        self.obs = np.zeros(self.num_obs_single_cfg * self.obs_history_len, dtype=np.float32)
        self.obs_buf_dict = {"actor_obs": self.obs[None, ...]}

        self._init_teleop()
    
    def _init_teleop(self):
        """Initialize teleoperation interface."""
        # Teleop (sim: keyboard; real: stick)
        self._kb_teleop = None
        self._stick_teleop = None
        self._keyboard_backend = "auto"

        if getattr(self.simulator, "is_real", False):
            stick = self.teleop_cfg.get("real_stick", {}) if isinstance(self.teleop_cfg, dict) else {}
            self._stick_teleop = RealStickTeleop(
                vx_scale=float(stick.get("vx_scale", 0.6)),
                vy_scale=float(stick.get("vy_scale", 0.6)),
                yaw_scale=float(stick.get("yaw_scale", 0.6)),
                smoothing=float(stick.get("smoothing", 0.5)),
            )
        else:
            kb = self.teleop_cfg.get("keyboard", {}) if isinstance(self.teleop_cfg, dict) else {}
            self._keyboard_backend = str(kb.get("backend", "auto"))
            step = kb.get("step", {"vx": 0.1, "vy": 0.1, "yaw": 0.1})
            limits = kb.get(
                "limits",
                {"vx": [-0.8, 1.2], "vy": [-0.6, 0.6], "yaw": [-0.8, 0.8]},
            )
            # use pygame keyboard control if available
            if self._keyboard_backend in ("auto", "pygame"):
                try:
                    cmd_screen = kb.get("command_screen_size", [360, 50])
                    keyboard_step = kb.get(
                        "pygame_step",
                        {"vx": 0.1, "vy": 0.1, "yaw": 0.1, "height": 0.1},
                    )
                    keyboard_limits = kb.get(
                        "pygame_limits",
                        {"vx": [-0.8, 1.2], "vy": [-0.6, 0.6], "yaw": [-0.8, 0.8], "height": [-0.5, 0.0]},
                    )
                    height_limits = kb.get("height_limits", [0.25, 0.75])
                    self._kb_teleop = PygameKeyboardTeleop(
                        command_screen_size=tuple(cmd_screen),
                        keyboard_step=(
                            float(keyboard_step["vx"]),
                            float(keyboard_step["vy"]),
                            float(keyboard_step["yaw"]),
                            float(keyboard_step["height"]),
                        ),
                        keyboard_limits=(
                            tuple(keyboard_limits["vx"]),
                            tuple(keyboard_limits["vy"]),
                            tuple(keyboard_limits["yaw"]),
                            tuple(keyboard_limits["height"]),
                        ),
                        base_height_cmd=float(self.height_cmd),
                        height_limits=tuple(height_limits),
                    )
                    self._keyboard_backend = "pygame"
                except Exception as exc:
                    if self._keyboard_backend == "pygame":
                        raise
                    logger.warning("pygame keyboard control unavailable, falling back to terminal: {}", exc)
                    self._keyboard_backend = "terminal"

            if self._kb_teleop is None:
                self._kb_teleop = TerminalKeyboardTeleop(
                    step=(float(step["vx"]), float(step["vy"]), float(step["yaw"])),
                    limits=(tuple(limits["vx"]), tuple(limits["vy"]), tuple(limits["yaw"])),
                    print_help=bool(kb.get("print_help", True)),
                )
                self._keyboard_backend = "terminal"

        logger.info(
            "LevelLocomotion initialized | actions: {}, single observation dimension: {}, history length: {}",
            self.num_actions,
            self.num_obs_single_cfg,
            self.obs_history_len,
        )
        if not getattr(self.simulator, "is_real", False):
            logger.info("Teleop backend (sim): {}", self._keyboard_backend)

    def _reset_history(self):
        self.obs_history.clear()
        for _ in range(self.obs_history_len):
            self.obs_history.append(np.zeros(self.num_obs_single_cfg, dtype=np.float32))

    # ------------------------------------------------------------------ #
    # Override basic process
    # ------------------------------------------------------------------ #
    def _reset_envs(self, refresh):
        super()._reset_envs(refresh)
        self.gait_phase = self.gait_phase_init.copy()
        self.flag_stand = True
        self.command_lin_vel *= 0
        self.command_ang_vel *= 0
        self._reset_history()

    def compute_observation(self):
        # Update commands from teleop first, then update basic observation (root/dof/state)
        self._update_commands()
        super()._update_obs()

        cmd = np.array(
            [self.command_lin_vel[0, 0], self.command_lin_vel[0, 1], self.command_ang_vel[0, 0]],
            dtype=np.float32,
        )

        single_obs = self._build_single_obs(cmd, self.height_cmd)
        if self.single_obs_dim is None:
            self.single_obs_dim = single_obs.shape[0]
            if self.single_obs_dim != self.num_obs_single_cfg:
                logger.warning(
                    "the single observation dimension does not match the cfg setting: {}, actual: {}",
                    self.num_obs_single_cfg,
                    self.single_obs_dim,
                )

        self.obs_history.append(single_obs)

        # Concatenate history observations
        for i, hist_obs in enumerate(self.obs_history):
            start = i * self.single_obs_dim
            self.obs[start : start + self.single_obs_dim] = hist_obs

        self.obs_buf_dict = {"actor_obs": self.obs[None, ...]}

        # Update time and gait phase
        self._update_gait_phase(cmd)

    def _update_commands(self):
        # real robot: use remote controller sticks (already decoded in simulator.real_world.RealWorld)
        if getattr(self.simulator, "is_real", False):
            if self._stick_teleop is None:
                return
            cmd = self._stick_teleop.update_from_sim(self.simulator)
            self.set_command_velocity(cmd.vx, cmd.vy, cmd.yaw)
            return

        # simulation: terminal keyboard
        if self._kb_teleop is None:
            return
        cmd = self._kb_teleop.update()
        self.set_command_velocity(cmd.vx, cmd.vy, cmd.yaw)
        # if input contains height offset, update height_cmd
        if hasattr(cmd, "height"):
            try:
                base_h = float(self.locomotion_cfg.get("height_cmd", self.height_cmd))
                height_limits = self.teleop_cfg.get("keyboard", {}).get("height_limits", [0.25, 0.75])
                self.height_cmd = float(np.clip(base_h + float(cmd.height), float(height_limits[0]), float(height_limits[1])))
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # LEVEL observation
    # ------------------------------------------------------------------ #
    def _build_single_obs(self, cmd: np.ndarray, height_cmd: float) -> np.ndarray:
        qj = self.dof_pos.reshape(-1)
        dqj = self.dof_vel.reshape(-1)
        omega = self.base_ang_vel.reshape(-1)

        # Base quaternion in framework is XYZW, need to convert to WXYZ and reuse LEVEL logic
        quat_xyzw = self.root_quat.reshape(-1)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32)
        gravity_orientation = quat_rotate_inverse(quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))

        # Default angles from current simulator (filtered by active degrees of freedom)
        default_angles = self.simulator.default_angles[self.simulator.active_dof_idx]
        if len(default_angles) < self.num_actions:
            padded = np.zeros(self.num_actions, dtype=np.float32)
            padded[: len(default_angles)] = default_angles
        else:
            padded = default_angles[: self.num_actions]

        qj_scaled = (qj - padded) * self.dof_pos_scale
        dqj_scaled = dqj * self.dof_vel_scale
        omega_scaled = omega * self.ang_vel_scale

        self.flag_stand = np.linalg.norm(cmd[:3]) < self.stand_threshold
        gait_sin = np.sin(2 * np.pi * self.gait_phase)

        action = self.action.reshape(-1)
        single_obs = np.zeros(self.num_obs_single_cfg, dtype=np.float32)
        single_obs[0:3] = cmd * self.cmd_scale
        single_obs[3] = height_cmd
        single_obs[4:7] = omega_scaled
        single_obs[7:10] = gravity_orientation
        single_obs[10 : 10 + self.num_actions] = qj_scaled[: self.num_actions]
        single_obs[10 + self.num_actions : 10 + 2 * self.num_actions] = dqj_scaled[: self.num_actions]
        single_obs[10 + 2 * self.num_actions : 10 + 3 * self.num_actions] = action[:self.num_actions]
        single_obs[10 + 2 * self.num_actions + 12 : 12 + 3 * self.num_actions] = gait_sin
        return single_obs

    def _update_gait_phase(self, cmd: np.ndarray):
        gait_dt = self.simulator.low_dt * self.simulator.decimation
        self.gait_phase = np.remainder(self.gait_phase + gait_dt * self.gait_period, 1.0).astype(np.float32)

        if self.flag_stand and np.any(np.abs(self.gait_phase - 0.38) < 0.05):
            self.gait_phase = np.array([0.38, 0.38], dtype=np.float32)
        elif (not self.flag_stand) and np.all(np.abs(self.gait_phase - 0.38) < 0.05):
            self.gait_phase = np.array([0.38, 0.88], dtype=np.float32)

