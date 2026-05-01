"""
Virtual Mouse — Final App (with Scroll)
========================================
Gestures:
  ✋  Move index finger              → move cursor
  🤏  Pinch index + thumb            → LEFT CLICK
  🤙  Pinch thumb + middle           → RIGHT CLICK
  ☝️✌️  Index + middle tips together   → SCROLL MODE
        then move hand UP             → scroll up
        then move hand DOWN           → scroll down
  👌  Pinch index+thumb IN/OUT      → Volume DOWN / UP  (press V first)

Keys:
  Q  → Quit
  D  → Toggle debug panel
  C  → Toggle control zone
  V  → Toggle volume mode
  F  → Toggle fullscreen

Install:
  pip install opencv-python mediapipe pyautogui
  # Windows volume: pip install pycaw comtypes
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

WEBCAM_INDEX  = 0
FRAME_WIDTH   = 1280
FRAME_HEIGHT  = 720

CTRL_X_MIN, CTRL_X_MAX = 0.12, 0.88
CTRL_Y_MIN, CTRL_Y_MAX = 0.08, 0.92

ALPHA_MIN   = 0.05
ALPHA_MAX   = 0.30
SPEED_SCALE = 0.005
DEAD_ZONE_PX = 2

PINCH_THRESHOLD  = 0.055   # thumb+index  → left click
RCLICK_THRESHOLD = 0.055   # thumb+middle → right click
SCROLL_THRESHOLD = 0.060   # index+middle → scroll mode

CLICK_COOLDOWN  = 0.55
CLICK_FLASH_DUR = 0.25

SCROLL_SENSITIVITY = 10    # higher = faster scrolling
SCROLL_DEAD_ZONE   = 0.006

VOLUME_SENSITIVITY = 150
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
    ctrl_x_min=CTRL_X_MIN,     ctrl_x_max=CTRL_X_MAX,
    ctrl_y_min=CTRL_Y_MIN,     ctrl_y_max=CTRL_Y_MAX,
    alpha_min=ALPHA_MIN,       alpha_max=ALPHA_MAX,
    speed_scale=SPEED_SCALE,
    dead_zone_px=DEAD_ZONE_PX,
    pinch_threshold=PINCH_THRESHOLD,
    rclick_threshold=RCLICK_THRESHOLD,
    scroll_threshold=SCROLL_THRESHOLD,
    click_cooldown=CLICK_COOLDOWN,
    click_flash_duration=CLICK_FLASH_DUR,
    scroll_sensitivity=SCROLL_SENSITIVITY,
    scroll_dead_zone=SCROLL_DEAD_ZONE,
    volume_sensitivity=VOLUME_SENSITIVITY,
    volume_dead_zone=VOLUME_DEAD_ZONE,
)

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)

WIN_NAME = "Virtual Mouse"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WIN_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# UI state
show_debug        = True
show_control_zone = True
volume_mode       = False
fullscreen        = True
prev_time         = time.time()

print("=" * 56)
print("  Virtual Mouse — with Scroll Gesture")
print(f"  Screen : {controller.screen_w} x {controller.screen_h}")
print("─" * 56)
print("  Gestures:")
print("    Index finger alone       → move cursor")
print("    Pinch (index + thumb)    → LEFT CLICK")
print("    Pinch (thumb + middle)   → RIGHT CLICK")
print("    Index + middle together  → SCROLL MODE")
print("      hand UP                → scroll up")
print("      hand DOWN              → scroll down")
print("    Pinch in/out [V mode]    → Volume ↓ / ↑")
print("─" * 56)
print("  Q=Quit  D=Debug  C=Zone  V=Volume  F=Fullscreen")
print("=" * 56)


# ═════════════════════════════════════════════════════════════════════════════
#  DRAW HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def draw_control_zone(frame, fw, fh):
    x1 = int(CTRL_X_MIN * fw);  y1 = int(CTRL_Y_MIN * fh)
    x2 = int(CTRL_X_MAX * fw);  y2 = int(CTRL_Y_MAX * fh)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (60, 60, 220), -1)
    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 100, 255), 1)
    cv2.putText(frame, "Control Zone", (x1 + 6, y1 - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 255), 1)


def draw_volume_bar(frame, fw, fh, volume):
    bar_h  = int(fh * 0.5)
    bar_x  = fw - 38
    bar_y  = (fh - bar_h) // 2
    fill_h = int(bar_h * volume / 100)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h), (50, 50, 50), -1)
    fill_color = (0, 220, 120) if volume > 20 else (0, 100, 220)
    cv2.rectangle(frame,
                  (bar_x, bar_y + bar_h - fill_h),
                  (bar_x + 20, bar_y + bar_h), fill_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 20, bar_y + bar_h), (180, 180, 180), 1)
    cv2.putText(frame, f"{volume}%", (bar_x - 10, bar_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    cv2.putText(frame, "VOL", (bar_x - 2, bar_y + bar_h + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 160), 1)


def draw_scroll_flash(frame, fw, fh, direction):
    """Big animated arrow + label when scrolling."""
    cx = fw // 2
    cy = fh // 2
    if direction < 0:   # scroll up
        cv2.arrowedLine(frame, (cx, cy + 30), (cx, cy - 50),
                        (80, 200, 255), 4, tipLength=0.35)
        cv2.putText(frame, "SCROLL UP", (cx - 80, cy - 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 255), 3)
    elif direction > 0:  # scroll down
        cv2.arrowedLine(frame, (cx, cy - 30), (cx, cy + 50),
                        (80, 200, 255), 4, tipLength=0.35)
        cv2.putText(frame, "SCROLL DOWN", (cx - 100, cy + 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 200, 255), 3)


def draw_debug_panel(frame, fw, fps):
    lines = [
        f"FPS          : {fps:.1f}",
        f"Cursor       : {controller.cursor_pos}",
        f"L-pinch      : {detector.pinch_distance:.3f}  thr={PINCH_THRESHOLD}",
        f"R-pinch      : {detector.rclick_distance:.3f}  thr={RCLICK_THRESHOLD}",
        f"Scroll dist  : {detector.scroll_distance:.3f}  thr={SCROLL_THRESHOLD}",
        f"Volume       : {controller.volume}%",
        f"Vol mode     : {'ON (V to off)' if volume_mode else 'OFF (V to on)'}",
    ]
    panel_w  = 320
    panel_h  = len(lines) * 22 + 14
    overlay  = frame.copy()
    cv2.rectangle(overlay, (fw - panel_w - 10, 6),
                  (fw - 8, panel_h + 6), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (fw - panel_w, 28 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (220, 220, 60), 1)


def draw_gesture_legend(frame, fw, fh, scroll_active, l_active, r_active):
    """Bottom row: highlight the currently active gesture."""
    items = [
        ("[IDX] Move",    (200, 200, 200),   True),
        ("[L-PINCH] LClick", (0, 255, 100),  l_active),
        ("[R-PINCH] RClick", (0, 180, 255),  r_active),
        ("[✌ SCROLL]",   (80, 200, 255),    scroll_active),
    ]
    x = 12
    for label, color, active in items:
        c = color if active else (100, 100, 100)
        cv2.putText(frame, label, (x, fh - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, c, 1)
        x += 185


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

scroll_direction = 0   # -1 up / +1 down / 0 none — used for flash overlay

while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Frame read failed.")
        break

    frame     = cv2.flip(frame, 1)
    fh, fw, _ = frame.shape

    # ── Detect ────────────────────────────────────────────────────────────────
    detector.process_frame(frame)

    if show_control_zone:
        draw_control_zone(frame, fw, fh)

    scroll_direction  = 0
    scroll_mode_active = False

    if detector.hand_detected:
        detector.draw_landmarks(frame)
        detector.draw_tip_dot(frame, detector.index_tip, radius=12, color=(0, 0, 255))

        # ── Determine active gesture (priority: scroll > click > move) ────────
        # Scroll mode: index + middle tips together
        scroll_active = detector.scroll_distance < SCROLL_THRESHOLD

        if scroll_active:
            # ── SCROLL MODE ───────────────────────────────────────────────────
            scroll_mode_active = True
            scroll_direction = controller.try_scroll(
                detector.scroll_distance,
                detector.index_y_norm,
                detector.prev_index_y_norm,
            )
            # Draw scroll indicator with direction arrow
            detector.draw_scroll_indicator(
                frame,
                detector.index_tip, detector.middle_tip,
                SCROLL_THRESHOLD,
                scroll_active,
                scroll_direction,
            )

        else:
            # ── CURSOR MODE ───────────────────────────────────────────────────
            controller.move(detector.index_tip.x, detector.index_tip.y)

            if volume_mode:
                # Pinch in/out → volume
                vol = controller.update_volume(
                    detector.pinch_distance,
                    detector.prev_pinch_distance,
                )
                delta = detector.pinch_distance - detector.prev_pinch_distance
                if abs(delta) > VOLUME_DEAD_ZONE:
                    tag = "▲ VOL UP" if delta > 0 else "▼ VOL DOWN"
                    cv2.putText(frame, tag, (fw // 2 - 80, fh // 2 - 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 220, 120), 3)
            else:
                # Left click: index + thumb
                detector.draw_gesture_line(
                    frame, detector.thumb_tip, detector.index_tip,
                    PINCH_THRESHOLD, "L-Click", (0, 255, 100))
                # Right click: thumb + middle
                detector.draw_gesture_line(
                    frame, detector.thumb_tip, detector.middle_tip,
                    RCLICK_THRESHOLD, "R-Click", (0, 180, 255))

                if controller.try_left_click(detector.pinch_distance):
                    print(f"  LEFT CLICK  at {controller.cursor_pos}")
                if controller.try_right_click(detector.rclick_distance):
                    print(f"  RIGHT CLICK at {controller.cursor_pos}")

        status_text  = "SCROLL MODE" if scroll_mode_active else "Cursor Mode"
        status_color = (80, 200, 255) if scroll_mode_active else (0, 220, 0)

    else:
        status_text  = "No hand detected"
        status_color = (0, 100, 255)

    # ── Flash overlays ────────────────────────────────────────────────────────
    if controller.lclick_flash_active:
        cv2.putText(frame, "LEFT CLICK!", (fw // 2 - 120, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 180), 4)
    if controller.rclick_flash_active:
        cv2.putText(frame, "RIGHT CLICK!", (fw // 2 - 140, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 180, 255), 4)
    if scroll_direction != 0:
        draw_scroll_flash(frame, fw, fh, scroll_direction)

    # ── Volume bar ────────────────────────────────────────────────────────────
    draw_volume_bar(frame, fw, fh, controller.volume)

    # ── FPS + debug ───────────────────────────────────────────────────────────
    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    if show_debug:
        draw_debug_panel(frame, fw, fps)

    # ── Status + footer ───────────────────────────────────────────────────────
    mode_tag = "  [VOL MODE — V to off]" if volume_mode else ""
    cv2.putText(frame, status_text + mode_tag, (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)
    cv2.putText(frame, "Q=Quit  D=Debug  C=Zone  V=Volume  F=Fullscreen",
                (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    # ── Gesture legend ────────────────────────────────────────────────────────
    draw_gesture_legend(
        frame, fw, fh,
        scroll_mode_active,
        detector.pinch_distance  < PINCH_THRESHOLD,
        detector.rclick_distance < RCLICK_THRESHOLD,
    )

    # ── Render ────────────────────────────────────────────────────────────────
    cv2.imshow(WIN_NAME, frame)

    # ── Keys ──────────────────────────────────────────────────────────────────
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