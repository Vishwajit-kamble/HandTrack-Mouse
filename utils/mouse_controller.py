"""
utils/mouse_controller.py
=========================
Cursor movement + click logic, fully decoupled from MediaPipe.
Handles: coordinate remapping, adaptive EMA smoothing,
         dead zone, pinch-to-click with cooldown.
"""

import pyautogui
import math
import time


class MouseController:
    """
    Takes normalized (0–1) landmark coordinates and drives the OS cursor.

    Usage:
        mc = MouseController()
        mc.move(lm_x, lm_y)          # call every frame
        mc.try_click(pinch_distance)  # call every frame
    """

    def __init__(
        self,
        ctrl_x_min=0.15, ctrl_x_max=0.85,
        ctrl_y_min=0.10, ctrl_y_max=0.90,
        alpha_min=0.10,  alpha_max=0.45,
        speed_scale=0.008,
        dead_zone_px=3,
        pinch_threshold=0.052,
        click_cooldown=0.5,
        click_flash_duration=0.2,
    ):
        # Control zone: the region of the webcam that maps to the full screen
        self.ctrl_x_min = ctrl_x_min;  self.ctrl_x_max = ctrl_x_max
        self.ctrl_y_min = ctrl_y_min;  self.ctrl_y_max = ctrl_y_max

        # Adaptive EMA smoothing parameters
        self.alpha_min   = alpha_min
        self.alpha_max   = alpha_max
        self.speed_scale = speed_scale

        # Dead zone: ignore sub-pixel tremor
        self.dead_zone_px = dead_zone_px

        # Click settings
        self.pinch_threshold    = pinch_threshold
        self.click_cooldown     = click_cooldown
        self.click_flash_dur    = click_flash_duration

        # Runtime state
        self.screen_w, self.screen_h = pyautogui.size()
        self.smooth_x    = self.screen_w / 2
        self.smooth_y    = self.screen_h / 2
        self._last_click = 0.0
        self._click_time = 0.0   # timestamp of last click (for flash)

        pyautogui.FAILSAFE = False
        pyautogui.PAUSE    = 0

    # ── Public API ────────────────────────────────────────────────────────────

    def move(self, norm_x, norm_y):
        """
        Given normalized landmark position (0–1), move the OS cursor.
        Applies control-zone remapping, adaptive smoothing, and dead zone.
        """
        target_x = self._remap(norm_x, self.ctrl_x_min, self.ctrl_x_max,
                                0, self.screen_w)
        target_y = self._remap(norm_y, self.ctrl_y_min, self.ctrl_y_max,
                                0, self.screen_h)

        alpha   = self._adaptive_alpha(target_x, target_y)
        new_x   = alpha * target_x + (1 - alpha) * self.smooth_x
        new_y   = alpha * target_y + (1 - alpha) * self.smooth_y

        delta = math.hypot(new_x - self.smooth_x, new_y - self.smooth_y)
        if delta > self.dead_zone_px:
            self.smooth_x = new_x
            self.smooth_y = new_y
            pyautogui.moveTo(int(self.smooth_x), int(self.smooth_y))

    def try_click(self, pinch_distance):
        """
        Fire a left-click if pinch_distance < threshold and cooldown has passed.
        Returns True if a click was fired this frame.
        """
        now = time.time()
        if pinch_distance < self.pinch_threshold and \
                (now - self._last_click) > self.click_cooldown:
            pyautogui.click()
            self._last_click = now
            self._click_time = now
            return True
        return False

    @property
    def is_pinching(self):
        return False  # stateless helper — caller passes pinch_distance

    @property
    def click_flash_active(self):
        """True for a short window after a click fires (for visual feedback)."""
        return (time.time() - self._click_time) < self.click_flash_dur

    @property
    def cursor_pos(self):
        return int(self.smooth_x), int(self.smooth_y)

    # ── Internals ─────────────────────────────────────────────────────────────

    def _remap(self, value, in_min, in_max, out_min, out_max):
        value = max(in_min, min(in_max, value))
        ratio = (value - in_min) / (in_max - in_min)
        return out_min + ratio * (out_max - out_min)

    def _adaptive_alpha(self, target_x, target_y):
        speed = math.hypot(target_x - self.smooth_x, target_y - self.smooth_y)
        return self.alpha_min + (self.alpha_max - self.alpha_min) * \
               min(1.0, speed * self.speed_scale)