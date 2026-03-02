import atexit
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TeleopCommand:
    vx: float = 0.0
    vy: float = 0.0
    yaw: float = 0.0
    height: float = 0.0
    policy_switch: Optional[str] = None  # "locomotion" or "mimic"


class _TerminalNonBlocking:
    """read non-blocking key from terminal"""

    def __init__(self):
        self._enabled = False
        self._old_termios = None
        self._old_flags: Optional[int] = None

    def enable(self):
        if self._enabled:
            return
        if not sys.stdin.isatty():
            return

        import termios
        import tty
        import fcntl
        import os

        fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        self._old_flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        fcntl.fcntl(fd, fcntl.F_SETFL, self._old_flags | os.O_NONBLOCK)
        self._enabled = True

    def disable(self):
        if not self._enabled:
            return
        if not sys.stdin.isatty():
            return

        import termios
        import fcntl

        fd = sys.stdin.fileno()
        if self._old_termios is not None:
            termios.tcsetattr(fd, termios.TCSADRAIN, self._old_termios)
        if self._old_flags is not None:
            fcntl.fcntl(fd, fcntl.F_SETFL, self._old_flags)
        self._enabled = False

    def read_key(self) -> Optional[str]:
        if not self._enabled:
            return None
        try:
            ch = sys.stdin.read(1)
            return ch if ch else None
        except Exception:
            return None


class TerminalKeyboardTeleop:
    """
    Simulation terminal keyboard control (does not depend on pygame / viewer callback).

    Default keys:
    - W/S: vx +/- step
    - A/D: vy +/- step
    - Q/E: yaw +/- step
    - Space: reset (vx, vy, yaw)
    - L: switch to locomotion policy
    - K: switch to mimic policy
    """

    def __init__(
        self,
        step: Tuple[float, float, float] = (0.1, 0.1, 0.1),
        limits: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]] = (
            (-0.8, 1.2),
            (-0.6, 0.6),
            (-0.8, 0.8),
        ),
        print_help: bool = True,
    ):
        self.step_vx, self.step_vy, self.step_yaw = step
        self.lim_vx, self.lim_vy, self.lim_yaw = limits
        self.cmd = TeleopCommand()
        self._term = _TerminalNonBlocking()
        self._term.enable()
        atexit.register(self.close)
        self._last_print = 0.0

        if print_help:
            sys.stderr.write(
                "\n[Teleop] Terminal keyboard enabled.\n"
                "  W/S: vx +/-\n"
                "  A/D: vy +/-\n"
                "  Q/E: yaw +/-\n"
                "  Space: stop\n"
                "  L: locomotion policy\n"
                "  K: mimic policy\n"
                "  Ctrl+C: exit\n\n"
            )
            sys.stderr.flush()

    def close(self):
        self._term.disable()

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def update(self) -> TeleopCommand:
        # Reset policy_switch flag at beginning of each update
        self.cmd.policy_switch = None
        
        # read all pending characters, apply in order
        while True:
            key = self._term.read_key()
            if key is None:
                break
            k = key.lower()
            if k == "w":
                self.cmd.vx += self.step_vx
            elif k == "s":
                self.cmd.vx -= self.step_vx
            elif k == "a":
                self.cmd.vy += self.step_vy
            elif k == "d":
                self.cmd.vy -= self.step_vy
            elif k == "q":
                self.cmd.yaw += self.step_yaw
            elif k == "e":
                self.cmd.yaw -= self.step_yaw
            elif k == "l":
                self.cmd.policy_switch = "locomotion"
            elif k == "k":
                self.cmd.policy_switch = "mimic"
            elif key == " ":
                self.cmd = TeleopCommand()

        self.cmd.vx = self._clip(self.cmd.vx, *self.lim_vx)
        self.cmd.vy = self._clip(self.cmd.vy, *self.lim_vy)
        self.cmd.yaw = self._clip(self.cmd.yaw, *self.lim_yaw)

        # reduce frequency of printing, avoid screen flickering
        now = time.time()
        if now - self._last_print > 0.5:
            self._last_print = now
            sys.stderr.write(
                f"\r[Teleop] vx={self.cmd.vx:+.2f} vy={self.cmd.vy:+.2f} yaw={self.cmd.yaw:+.2f}    "
            )
            sys.stderr.flush()

        return self.cmd


class PygameKeyboardTeleop:
    """
    - direction keys: vx/vy
    - N/M: yaw
    - Space: stop
    - H/J: height (only for level_locomotion)
    - L: switch to locomotion policy
    - K: switch to mimic policy
    """

    def __init__(
        self,
        command_screen_size=(360, 50),
        keyboard_step=(0.1, 0.1, 0.1, 0.1),  # vx, vy, yaw, height
        keyboard_limits=((-0.8, 1.2), (-0.6, 0.6), (-0.8, 0.8), (-0.5, 0.0)),
        base_height_cmd: float = 0.75,
        height_limits=(0.25, 0.75),
    ):
        try:
            import pygame  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"pygame not available: {exc}") from exc

        self._pygame = pygame
        self.cmd = TeleopCommand()
        self.base_height_cmd = float(base_height_cmd)
        self.height_limits = (float(height_limits[0]), float(height_limits[1]))

        self.step_vx, self.step_vy, self.step_yaw, self.step_h = map(float, keyboard_step)
        (self.lim_vx, self.lim_vy, self.lim_yaw, self.lim_h) = keyboard_limits

        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode(tuple(command_screen_size))
        pygame.display.set_caption("Command")
        self.font = pygame.font.Font(None, 20)

        atexit.register(self.close)

    def close(self):
        try:
            self._pygame.quit()
        except Exception:
            pass

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _handle_key(self, key):
        pygame = self._pygame
        if key == pygame.K_UP:
            self.cmd.vx += self.step_vx
        elif key == pygame.K_DOWN:
            self.cmd.vx -= self.step_vx
        elif key == pygame.K_LEFT:
            self.cmd.vy += self.step_vy
        elif key == pygame.K_RIGHT:
            self.cmd.vy -= self.step_vy
        elif key == pygame.K_n:
            self.cmd.yaw += self.step_yaw
        elif key == pygame.K_m:
            self.cmd.yaw -= self.step_yaw
        elif key == pygame.K_h:
            self.cmd.height += self.step_h
        elif key == pygame.K_j:
            self.cmd.height -= self.step_h
        elif key == pygame.K_l:
            self.cmd.policy_switch = "locomotion"
        elif key == pygame.K_k:
            self.cmd.policy_switch = "mimic"
        elif key == pygame.K_SPACE:
            self.cmd = TeleopCommand()

    def update(self) -> TeleopCommand:
        # Reset policy_switch flag at beginning of each update
        self.cmd.policy_switch = None
        
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.cmd = TeleopCommand()
            if event.type == pygame.KEYDOWN:
                self._handle_key(event.key)

        self.cmd.vx = self._clip(self.cmd.vx, float(self.lim_vx[0]), float(self.lim_vx[1]))
        self.cmd.vy = self._clip(self.cmd.vy, float(self.lim_vy[0]), float(self.lim_vy[1]))
        self.cmd.yaw = self._clip(self.cmd.yaw, float(self.lim_yaw[0]), float(self.lim_yaw[1]))
        self.cmd.height = self._clip(self.cmd.height, float(self.lim_h[0]), float(self.lim_h[1]))

        # render
        if self.screen is not None and self.font is not None:
            self.screen.fill((0, 0, 0))
            height_cmd = self._clip(
                self.base_height_cmd + self.cmd.height, self.height_limits[0], self.height_limits[1]
            )
            text = self.font.render(
                f"vx: {self.cmd.vx:.2f} vy: {self.cmd.vy:.2f} dyaw: {self.cmd.yaw:.2f} height: {height_cmd:.2f}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(text, (10, 15))
            pygame.display.update()

        return self.cmd


class RealStickTeleop:
    """
    Real robot stick control: directly use simulator's left_stick/right_stick with smoothing.
    
    Button mapping:
    - L1 (left_upper_switch): switch to locomotion policy
    - L2 (left_lower_right_switch): switch to mimic policy
    """

    def __init__(
        self,
        vx_scale: float = 1.0,
        vy_scale: float = 1.0,
        yaw_scale: float = 1.0,
        smoothing: float = 0.5,
    ):
        """
        Args:
            vx_scale: Scale factor for forward/backward velocity
            vy_scale: Scale factor for left/right velocity
            yaw_scale: Scale factor for yaw rotation
            smoothing: Smoothing factor (0-1), lower means more smoothing, 1.0 means no smoothing
        """
        self.vx_scale = float(vx_scale)
        self.vy_scale = float(vy_scale)
        self.yaw_scale = float(yaw_scale)
        self.smoothing = float(smoothing)
        
        # Cache previous command for smoothing
        self.prev_cmd = TeleopCommand(vx=0.0, vy=0.0, yaw=0.0)

    def update_from_sim(self, simulator) -> TeleopCommand:
        if not hasattr(simulator, "left_stick") or not hasattr(simulator, "right_stick"):
            return self.prev_cmd
        
        left = simulator.left_stick
        right = simulator.right_stick
        
        # Raw command from stick input
        raw_cmd = TeleopCommand(
            vx=float(left[1]) * self.vx_scale,
            vy=float(left[0]) * self.vy_scale,
            yaw=float(right[0]) * self.yaw_scale,
        )
        
        # Apply exponential smoothing: cmd = (1 - alpha) * prev + alpha * raw
        # smoothing=1 means no smoothing, smoothing=0 means full smoothing
        alpha = self.smoothing
        smoothed_cmd = TeleopCommand(
            vx=(1.0 - alpha) * self.prev_cmd.vx + alpha * raw_cmd.vx,
            vy=(1.0 - alpha) * self.prev_cmd.vy + alpha * raw_cmd.vy,
            yaw=(1.0 - alpha) * self.prev_cmd.yaw + alpha * raw_cmd.yaw,
        )
        
        # Check for policy switch button presses
        policy_switch = None
        if hasattr(simulator, "left_upper_switch_pressed") and simulator.left_upper_switch_pressed:
            policy_switch = "locomotion"
            simulator.left_upper_switch_pressed = False  # Clear the flag
        elif hasattr(simulator, "left_lower_left_switch_pressed") and simulator.left_lower_left_switch_pressed:
            policy_switch = "mimic"
            simulator.left_lower_left_switch_pressed = False  # Clear the flag
        
        smoothed_cmd.policy_switch = policy_switch
        
        # Update cache
        self.prev_cmd = smoothed_cmd
        
        return smoothed_cmd

