"""
utils/mouse_controller.py
=========================
Cursor movement, left-click, right-click, scroll, and volume control.
"""

import pyautogui
import math
import time
import sys
import subprocess


# ── Volume backend (cross-platform) ──────────────────────────────────────────

class _MacVolume:
    def get(self):
        try:
            out = subprocess.check_output(
                ["osascript", "-e", "output volume of (get volume settings)"])
            return int(out.strip())
        except Exception:
            return 50

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
                ["amixer", "-D", "pulse", "get", "Master"], text=True)
            import re
            m = re.search(r"(\d+)%", out)
            return int(m.group(1)) if m else 50
        except Exception:
            return 50

    def set(self, vol):
        vol = max(0, min(100, int(vol)))
        subprocess.call(["amixer", "-D", "pulse", "set", "Master", f"{vol}%"],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _get_volume_controller():
    if sys.platform == "darwin":
        return _MacVolume()
    elif sys.platform == "win32":
        return _WinVolume()
    else:
        return _LinuxVolume()


# ── MouseController ───────────────────────────────────────────────────────────

class MouseController:

    def __init__(
        self,
        ctrl_x_min=0.12, ctrl_x_max=0.88,
        ctrl_y_min=0.08, ctrl_y_max=0.92,
        alpha_min=0.05,  alpha_max=0.30,
        speed_scale=0.005,
        dead_zone_px=2,
        pinch_threshold=0.055,
        rclick_threshold=0.055,
        scroll_threshold=0.055,   # index ↔ middle distance to enter scroll mode
        click_cooldown=0.55,
        click_flash_duration=0.25,
        scroll_sensitivity=8,     # pyautogui scroll clicks per unit of hand movement
        scroll_dead_zone=0.008,   # minimum y-delta to trigger scroll
        volume_sensitivity=150,
        volume_dead_zone=0.003,
    ):
        self.ctrl_x_min = ctrl_x_min;  self.ctrl_x_max = ctrl_x_max
        self.ctrl_y_min = ctrl_y_min;  self.ctrl_y_max = ctrl_y_max
        self.alpha_min   = alpha_min
        self.alpha_max   = alpha_max
        self.speed_scale = speed_scale
        self.dead_zone_px = dead_zone_px

        self.pinch_threshold  = pinch_threshold
        self.rclick_threshold = rclick_threshold
        self.scroll_threshold = scroll_threshold
        self.click_cooldown   = click_cooldown
        self.click_flash_dur  = click_flash_duration

        self.scroll_sensitivity = scroll_sensitivity
        self.scroll_dead_zone   = scroll_dead_zone
        self._scroll_accumulator = 0.0   # accumulate small movements before firing

        self.vol_sensitivity = volume_sensitivity
        self.vol_dead_zone   = volume_dead_zone

        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_x  = self.screen_w / 2
        self.smooth_y  = self.screen_h / 2

        self._last_lclick  = 0.0
        self._last_rclick  = 0.0
        self._lclick_time  = 0.0
        self._rclick_time  = 0.0
        self._last_scroll_dir = 0   # -1 up, +1 down, 0 none

        self._vol_ctrl = _get_volume_controller()
        self._volume   = self._vol_ctrl.get()

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0

    # ── Cursor ────────────────────────────────────────────────────────────────

    def move(self, norm_x, norm_y):
        tx = self._remap(norm_x, self.ctrl_x_min, self.ctrl_x_max, 0, self.screen_w)
        ty = self._remap(norm_y, self.ctrl_y_min, self.ctrl_y_max, 0, self.screen_h)
        alpha = self._adaptive_alpha(tx, ty)
        new_x = alpha * tx + (1 - alpha) * self.smooth_x
        new_y = alpha * ty + (1 - alpha) * self.smooth_y
        delta = math.hypot(new_x - self.smooth_x, new_y - self.smooth_y)
        if delta > self.dead_zone_px:
            self.smooth_x = new_x
            self.smooth_y = new_y
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

    # ── Click ─────────────────────────────────────────────────────────────────

    def try_left_click(self, pinch_dist):
        now = time.time()
        if pinch_dist < self.pinch_threshold and \
                (now - self._last_lclick) > self.click_cooldown:
            pyautogui.click(button='left')
            self._last_lclick = now
            self._lclick_time = now
            return True
        return False

    def try_right_click(self, rclick_dist):
        now = time.time()
        if rclick_dist < self.rclick_threshold and \
                (now - self._last_rclick) > self.click_cooldown:
            pyautogui.click(button='right')
            self._last_rclick = now
            self._rclick_time = now
            return True
        return False

    # ── Scroll ────────────────────────────────────────────────────────────────

    def try_scroll(self, scroll_dist, current_y, prev_y):
        """
        Call every frame while index+middle tips are close.
        scroll_dist  : current normalized distance between index and middle tips
        current_y    : current normalized y of index tip (0=top, 1=bottom)
        prev_y       : previous frame normalized y of index tip

        Returns scroll direction: -1 (up), +1 (down), 0 (none)
        """
        if scroll_dist > self.scroll_threshold:
            self._scroll_accumulator = 0.0
            self._last_scroll_dir = 0
            return 0

        # y increases downward in normalized space
        # hand moving up → y decreases → delta negative → scroll UP (positive pyautogui)
        delta_y = current_y - prev_y   # negative = hand moved up

        if abs(delta_y) < self.scroll_dead_zone:
            self._last_scroll_dir = 0
            return 0

        # Accumulate movement, fire scroll ticks
        self._scroll_accumulator += delta_y * self.scroll_sensitivity

        ticks = int(self._scroll_accumulator)
        if ticks != 0:
            # pyautogui.scroll: positive = scroll up, negative = scroll down
            pyautogui.scroll(-ticks)   # invert: hand up → scroll up
            self._scroll_accumulator -= ticks

        direction = -1 if delta_y < 0 else 1   # -1=up, +1=down
        self._last_scroll_dir = direction
        return direction

    # ── Volume ────────────────────────────────────────────────────────────────

    def update_volume(self, current_pinch, prev_pinch):
        delta = current_pinch - prev_pinch
        if abs(delta) > self.vol_dead_zone:
            change = delta * self.vol_sensitivity
            self._volume = max(0, min(100, self._volume + change))
            self._vol_ctrl.set(self._volume)
        return self._volume

    # ── Flash states ──────────────────────────────────────────────────────────

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