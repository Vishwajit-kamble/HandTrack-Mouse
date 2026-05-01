"""
utils/mouse_controller.py
=========================
Cursor movement, left-click, right-click, and system volume control.
All decoupled from MediaPipe — receives normalized coords and distances.
"""

import pyautogui
import math
import time
import sys
import subprocess


# ── Volume backend (cross-platform) ──────────────────────────────────────────

def _get_volume_controller():
    """Return a platform-appropriate volume controller."""
    platform = sys.platform

    if platform == "darwin":          # macOS
        return _MacVolume()
    elif platform == "win32":         # Windows
        return _WinVolume()
    else:                             # Linux
        return _LinuxVolume()


class _MacVolume:
    def get(self):
        out = subprocess.check_output(
            ["osascript", "-e", "output volume of (get volume settings)"]
        )
        return int(out.strip())

    def set(self, vol):
        vol = max(0, min(100, int(vol)))
        subprocess.call(["osascript", "-e", f"set volume output volume {vol}"])


class _WinVolume:
    def __init__(self):
        try:
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self._vol = cast(interface, POINTER(IAudioEndpointVolume))
        except Exception:
            self._vol = None

    def get(self):
        if self._vol:
            return int(self._vol.GetMasterVolumeLevelScalar() * 100)
        return 50

    def set(self, vol):
        if self._vol:
            self._vol.SetMasterVolumeLevelScalar(max(0, min(100, vol)) / 100, None)


class _LinuxVolume:
    def get(self):
        try:
            out = subprocess.check_output(
                ["amixer", "-D", "pulse", "get", "Master"], text=True
            )
            import re
            m = re.search(r"(\d+)%", out)
            return int(m.group(1)) if m else 50
        except Exception:
            return 50

    def set(self, vol):
        vol = max(0, min(100, int(vol)))
        subprocess.call(["amixer", "-D", "pulse", "set", "Master", f"{vol}%"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ── MouseController ───────────────────────────────────────────────────────────

class MouseController:
    """
    Drives the OS cursor with smooth adaptive movement, left-click,
    right-click, and pinch-based volume control.
    """

    def __init__(
        self,
        ctrl_x_min=0.15, ctrl_x_max=0.85,
        ctrl_y_min=0.10, ctrl_y_max=0.90,
        alpha_min=0.06,  alpha_max=0.35,
        speed_scale=0.006,
        dead_zone_px=2,
        pinch_threshold=0.052,
        rclick_threshold=0.052,
        click_cooldown=0.55,
        rclick_cooldown=0.55,
        click_flash_duration=0.25,
        volume_sensitivity=120,      # higher = faster volume change per px of pinch
        volume_dead_zone=0.004,      # minimum pinch delta to count as volume gesture
    ):
        self.ctrl_x_min = ctrl_x_min;  self.ctrl_x_max = ctrl_x_max
        self.ctrl_y_min = ctrl_y_min;  self.ctrl_y_max = ctrl_y_max
        self.alpha_min   = alpha_min
        self.alpha_max   = alpha_max
        self.speed_scale = speed_scale
        self.dead_zone_px = dead_zone_px

        self.pinch_threshold    = pinch_threshold
        self.rclick_threshold   = rclick_threshold
        self.click_cooldown     = click_cooldown
        self.rclick_cooldown    = rclick_cooldown
        self.click_flash_dur    = click_flash_duration
        self.vol_sensitivity    = volume_sensitivity
        self.vol_dead_zone      = volume_dead_zone

        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_x  = self.screen_w / 2
        self.smooth_y  = self.screen_h / 2

        self._last_lclick  = 0.0
        self._last_rclick  = 0.0
        self._lclick_time  = 0.0
        self._rclick_time  = 0.0

        self._vol_ctrl = _get_volume_controller()
        self._volume   = self._vol_ctrl.get()   # sync with current system volume

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0

    # ── Cursor movement ───────────────────────────────────────────────────────

    def move(self, norm_x, norm_y):
        """Map normalized (0–1) index-tip coords to screen and move cursor."""
        tx = self._remap(norm_x, self.ctrl_x_min, self.ctrl_x_max, 0, self.screen_w)
        ty = self._remap(norm_y, self.ctrl_y_min, self.ctrl_y_max, 0, self.screen_h)

        alpha  = self._adaptive_alpha(tx, ty)
        new_x  = alpha * tx + (1 - alpha) * self.smooth_x
        new_y  = alpha * ty + (1 - alpha) * self.smooth_y

        delta = math.hypot(new_x - self.smooth_x, new_y - self.smooth_y)
        if delta > self.dead_zone_px:
            self.smooth_x = new_x
            self.smooth_y = new_y
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

    # ── Click actions ─────────────────────────────────────────────────────────

    def try_left_click(self, pinch_dist):
        """Fire a left-click if pinch (index+thumb) close enough & cooldown passed."""
        now = time.time()
        if pinch_dist < self.pinch_threshold and \
                (now - self._last_lclick) > self.click_cooldown:
            pyautogui.click(button='left')
            self._last_lclick = now
            self._lclick_time = now
            return True
        return False

    def try_right_click(self, rclick_dist):
        """Fire a right-click if thumb+middle close enough & cooldown passed."""
        now = time.time()
        if rclick_dist < self.rclick_threshold and \
                (now - self._last_rclick) > self.rclick_cooldown:
            pyautogui.click(button='right')
            self._last_rclick = now
            self._rclick_time = now
            return True
        return False

    # ── Volume control ────────────────────────────────────────────────────────

    def update_volume(self, current_pinch, prev_pinch):
        """
        Compare current vs previous pinch distance.
        Pinch OUT (fingers apart → dist increases) → volume up.
        Pinch IN  (fingers close → dist decreases) → volume down.
        Returns the new volume level (0–100).
        """
        delta = current_pinch - prev_pinch   # positive = moving apart = louder

        if abs(delta) > self.vol_dead_zone:
            change = delta * self.vol_sensitivity
            self._volume = max(0, min(100, self._volume + change))
            self._vol_ctrl.set(self._volume)

        return self._volume

    # ── Flash state ───────────────────────────────────────────────────────────

    @property
    def lclick_flash_active(self):
        return (time.time() - self._lclick_time) < self.click_flash_dur

    @property
    def rclick_flash_active(self):
        return (time.time() - self._rclick_time) < self.click_flash_dur

    @property
    def cursor_pos(self):
        return int(self.smooth_x), int(self.smooth_y)

    @property
    def volume(self):
        return int(self._volume)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _remap(self, value, in_min, in_max, out_min, out_max):
        value = max(in_min, min(in_max, value))
        return out_min + ((value - in_min) / (in_max - in_min)) * (out_max - out_min)

    def _adaptive_alpha(self, tx, ty):
        speed = math.hypot(tx - self.smooth_x, ty - self.smooth_y)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * \
               min(1.0, speed * self.speed_scale)