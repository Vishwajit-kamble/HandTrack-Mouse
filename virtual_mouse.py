"""
Virtual Mouse — Upgraded Final App
===================================
Gestures:
  ✋ Move index finger          → move cursor (ultra-smooth)
  🤏 Pinch index + thumb        → LEFT CLICK
  🤙 Pinch thumb + middle       → RIGHT CLICK
  👌 Pinch index+thumb OUT/IN   → Volume UP / DOWN  (while holding pinch zone)

Controls:
  Q  → Quit
  D  → Toggle debug panel
  C  → Toggle control zone
  V  → Toggle volume mode (pinch = volume instead of click)
  F  → Toggle fullscreen webcam window

Install:
  pip install opencv-python mediapipe pyautogui
  # Windows only (volume): pip install pycaw comtypes
"""

import cv2
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hand_detector    import HandDetector
from utils.mouse_controller import MouseController

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

WEBCAM_INDEX   = 0
FRAME_WIDTH    = 1280    # higher res = more accurate tracking
FRAME_HEIGHT   = 720

# Control zone — inner region of frame that maps to full screen
# Wider zone = less arm movement needed
CTRL_X_MIN, CTRL_X_MAX = 0.12, 0.88
CTRL_Y_MIN, CTRL_Y_MAX = 0.08, 0.92

# Smoothing — lower alpha_max = smoother but slightly less snappy
ALPHA_MIN   = 0.05    # at rest: very stable
ALPHA_MAX   = 0.30    # fast swipe: responsive
SPEED_SCALE = 0.005

DEAD_ZONE_PX = 2      # ignore sub-2px tremor

# Gesture thresholds (normalized 0–1 distance)
PINCH_THRESHOLD  = 0.055   # index+thumb → left click
RCLICK_THRESHOLD = 0.055   # thumb+middle → right click

CLICK_COOLDOWN   = 0.55    # seconds between clicks
CLICK_FLASH_DUR  = 0.25

# Volume
VOLUME_SENSITIVITY = 150   # how fast volume changes per pinch delta
VOLUME_DEAD_ZONE   = 0.003

# ═════════════════════════════════════════════════════════════════════════════
#  INIT
# ═════════════════════════════════════════════════════════════════════════════

detector = HandDetector(
    max_hands=1,
    detection_confidence=0.75,
    tracking_confidence=0.6,
)

controller = MouseController(
    ctrl_x_min=CTRL_X_MIN, ctrl_x_max=CTRL_X_MAX,
    ctrl_y_min=CTRL_Y_MIN, ctrl_y_max=CTRL_Y_MAX,
    alpha_min=ALPHA_MIN,   alpha_max=ALPHA_MAX,
    speed_scale=SPEED_SCALE,
    dead_zone_px=DEAD_ZONE_PX,
    pinch_threshold=PINCH_THRESHOLD,
    rclick_threshold=RCLICK_THRESHOLD,
    click_cooldown=CLICK_COOLDOWN,
    rclick_cooldown=CLICK_COOLDOWN,
    click_flash_duration=CLICK_FLASH_DUR,
    volume_sensitivity=VOLUME_SENSITIVITY,
    volume_dead_zone=VOLUME_DEAD_ZONE,
)

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)   # request 60fps for smoother tracking

# UI state
show_debug        = True
show_control_zone = True
volume_mode       = False   # True = pinch controls volume instead of clicking
fullscreen        = True    # Start fullscreen

WIN_NAME = "Virtual Mouse"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
if fullscreen:
    cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

prev_time = time.time()

print("=" * 55)
print("  Virtual Mouse — Upgraded")
print(f"  Screen : {controller.screen_w} x {controller.screen_h}")
print("─" * 55)
print("  Gestures:")
print("    Index finger      → move cursor")
print("    Pinch (idx+thumb) → LEFT click")
print("    Pinch (thb+mid)   → RIGHT click")
print("    Pinch in/out      → Volume ↓ / ↑  (hold V)")
print("─" * 55)
print("  Keys: Q=quit  D=debug  C=zone  V=vol-mode  F=fullscreen")
print("=" * 55)


# ═════════════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def draw_control_zone(frame, fw, fh):
    x1 = int(CTRL_X_MIN * fw);  y1 = int(CTRL_Y_MIN * fh)
    x2 = int(CTRL_X_MAX * fw);  y2 = int(CTRL_Y_MAX * fh)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 220), -1)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)  # subtle fill
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)
    cv2.putText(frame, "Control Zone", (x1 + 6, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 255), 1)


def draw_volume_bar(frame, fw, fh, volume):
    """Draw a vertical volume bar on the right side of the frame."""
    bar_h   = int(fh * 0.5)
    bar_x   = fw - 38
    bar_y   = (fh - bar_h) // 2
    fill_h  = int(bar_h * volume / 100)

    # Background
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h),
                  (50, 50, 50), -1)
    # Fill
    fill_color = (0, 220, 120) if volume > 20 else (0, 100, 220)
    cv2.rectangle(frame, (bar_x, bar_y + bar_h - fill_h),
                  (bar_x + 20, bar_y + bar_h), fill_color, -1)
    # Border
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h),
                  (180, 180, 180), 1)
    cv2.putText(frame, f"{volume}%", (bar_x - 10, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(frame, "VOL", (bar_x - 2, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def draw_debug_panel(frame, fw, detector, controller, fps):
    lines = [
        f"FPS         : {fps:.1f}",
        f"Cursor      : {controller.cursor_pos}",
        f"L-pinch     : {detector.pinch_distance:.3f}  (thr {PINCH_THRESHOLD})",
        f"R-pinch     : {detector.rclick_distance:.3f}  (thr {RCLICK_THRESHOLD})",
        f"Volume      : {controller.volume}%",
        f"Vol mode    : {'ON' if volume_mode else 'OFF'}",
        f"Alpha range : {ALPHA_MIN}–{ALPHA_MAX}",
        f"Dead zone   : {DEAD_ZONE_PX}px",
    ]
    # semi-transparent background
    panel_w, panel_h = 310, len(lines) * 22 + 12
    overlay = frame.copy()
    cv2.rectangle(overlay, (fw - panel_w - 8, 8),
                  (fw - 8, panel_h + 8), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (fw - panel_w, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 60), 1)


def draw_gesture_hud(frame, fw, fh, l_pinch, r_pinch, vol_mode):
    """Bottom gesture legend."""
    gestures = [
        ("IDX", "Cursor",    (200, 200, 200)),
        ("L-PINCH", "Left Click" if not vol_mode else "Volume",
         (0, 220, 100) if l_pinch < PINCH_THRESHOLD else (180, 180, 180)),
        ("R-PINCH", "Right Click",
         (0, 180, 255) if r_pinch < RCLICK_THRESHOLD else (180, 180, 180)),
    ]
    y = fh - 18
    x = 16
    for key, label, color in gestures:
        cv2.putText(frame, f"[{key}]={label}", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        x += 160


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Frame read failed.")
        break

    frame   = cv2.flip(frame, 1)
    fh, fw, _ = frame.shape

    # ── Detect ────────────────────────────────────────────────────────────────
    detector.process_frame(frame)

    # ── Control zone ──────────────────────────────────────────────────────────
    if show_control_zone:
        draw_control_zone(frame, fw, fh)

    # ── Hand logic ────────────────────────────────────────────────────────────
    if detector.hand_detected:
        detector.draw_landmarks(frame)
        detector.draw_tip_dot(frame, detector.index_tip, radius=12, color=(0, 0, 255))

        # Always draw gesture lines so user sees feedback
        detector.draw_gesture_line(
            frame, detector.thumb_tip, detector.index_tip,
            PINCH_THRESHOLD, "L-Click" if not volume_mode else "Vol"
        )
        detector.draw_gesture_line(
            frame, detector.thumb_tip, detector.middle_tip,
            RCLICK_THRESHOLD, "R-Click"
        )

        # ── Move cursor ───────────────────────────────────────────────────
        controller.move(detector.index_tip.x, detector.index_tip.y)

        if volume_mode:
            # Pinch index+thumb in/out → volume
            vol = controller.update_volume(
                detector.pinch_distance,
                detector.prev_pinch_distance
            )
            direction = ""
            delta = detector.pinch_distance - detector.prev_pinch_distance
            if abs(delta) > VOLUME_DEAD_ZONE:
                direction = "▲ VOL UP" if delta > 0 else "▼ VOL DOWN"
            if direction:
                cv2.putText(frame, direction, (fw // 2 - 70, fh // 2 - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 3)
        else:
            # ── Left click: index + thumb ──────────────────────────────
            if controller.try_left_click(detector.pinch_distance):
                cx, cy = controller.cursor_pos
                print(f"  LEFT CLICK  at ({cx}, {cy})")

            # ── Right click: thumb + middle ────────────────────────────
            if controller.try_right_click(detector.rclick_distance):
                cx, cy = controller.cursor_pos
                print(f"  RIGHT CLICK at ({cx}, {cy})")

        status_text  = "Hand Detected"
        status_color = (0, 220, 0)
    else:
        status_text  = "No hand detected — show your hand"
        status_color = (0, 100, 255)

    # ── Click flash overlays ──────────────────────────────────────────────────
    if controller.lclick_flash_active:
        cv2.putText(frame, "LEFT CLICK!", (fw // 2 - 110, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 180), 4)
    if controller.rclick_flash_active:
        cv2.putText(frame, "RIGHT CLICK!", (fw // 2 - 130, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 180, 255), 4)

    # ── Volume bar ────────────────────────────────────────────────────────────
    draw_volume_bar(frame, fw, fh, controller.volume)

    # ── FPS ───────────────────────────────────────────────────────────────────
    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    # ── Debug panel ───────────────────────────────────────────────────────────
    if show_debug:
        draw_debug_panel(frame, fw, detector, controller, fps)

    # ── Status bar ────────────────────────────────────────────────────────────
    mode_tag = "  [VOL MODE]" if volume_mode else ""
    cv2.putText(frame, status_text + mode_tag, (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
    cv2.putText(frame, "Q=Quit  D=Debug  C=Zone  V=Volume  F=Fullscreen",
                (16, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)

    # ── Gesture HUD ───────────────────────────────────────────────────────────
    draw_gesture_hud(frame, fw, fh,
                     detector.pinch_distance, detector.rclick_distance, volume_mode)

    # ── Render ────────────────────────────────────────────────────────────────
    cv2.imshow(WIN_NAME, frame)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting.")
        break
    elif key == ord('d'):
        show_debug = not show_debug
    elif key == ord('c'):
        show_control_zone = not show_control_zone
    elif key == ord('v'):
        volume_mode = not volume_mode
        print(f"  Volume mode: {'ON' if volume_mode else 'OFF'}")
    elif key == ord('f'):
        fullscreen = not fullscreen
        prop = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
        cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, prop)

# ═════════════════════════════════════════════════════════════════════════════
#  CLEANUP
# ═════════════════════════════════════════════════════════════════════════════

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Done.")