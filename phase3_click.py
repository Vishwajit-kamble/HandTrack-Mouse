"""
Virtual Mouse - Phase 3: Smoothing + Pinch-to-Click
=====================================================
Goal: Add refined smoothing so the cursor feels stable and natural,
      plus pinch gesture detection (index tip + thumb tip close together)
      to trigger a real left-click via PyAutoGUI.

New in this phase:
  - Adaptive smoothing: cursor moves fast for big movements, smooth for small
  - Pinch-to-click: distance between landmark #4 (thumb) and #8 (index) < threshold
  - Click cooldown: prevents accidental double-clicks from a single pinch hold
  - Visual feedback: pinch indicator ring + "CLICK!" flash on screen
  - Dead zone: tiny movements below a pixel threshold are ignored (rock-solid hold)

Install dependencies:
    pip install opencv-python mediapipe pyautogui

Run:
    python phase3_click.py

Controls:
    Press 'q' to quit
    Pinch index finger + thumb to click
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# ── PyAutoGUI config ──────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0

SCREEN_W, SCREEN_H = pyautogui.size()
print(f"Screen: {SCREEN_W} x {SCREEN_H}")

# ── Control zone ──────────────────────────────────────────────────────────────
CTRL_X_MIN = 0.15
CTRL_X_MAX = 0.85
CTRL_Y_MIN = 0.10
CTRL_Y_MAX = 0.90

# ── Smoothing ─────────────────────────────────────────────────────────────────
# Adaptive EMA: alpha scales with how far the finger moved.
# Small move → low alpha (smooth). Big move → high alpha (responsive).
ALPHA_MIN   = 0.10   # minimum smoothing (very steady)
ALPHA_MAX   = 0.45   # maximum responsiveness (fast swipes)
SPEED_SCALE = 0.008  # how quickly alpha ramps up with speed

# Dead zone: if cursor would move less than this many pixels, don't move it.
# Eliminates the micro-drift when your hand is resting still.
DEAD_ZONE_PX = 3

# ── Pinch-to-click config ─────────────────────────────────────────────────────
# Distance between thumb tip (#4) and index tip (#8), in NORMALIZED units.
# Normalized space: 0.0–1.0 across the frame. ~0.05 ≈ fingertips touching.
PINCH_THRESHOLD  = 0.052   # closer than this = pinch detected
CLICK_COOLDOWN_S = 0.5     # seconds to wait before allowing next click
CLICK_FLASH_S    = 0.2     # how long the "CLICK!" text stays on screen

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles  = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# ── Webcam setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open webcam.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ── State ─────────────────────────────────────────────────────────────────────
smooth_x       = SCREEN_W / 2
smooth_y       = SCREEN_H / 2
last_click_t   = 0.0       # timestamp of last click (for cooldown)
click_flash_t  = 0.0       # timestamp of last click (for visual flash)
prev_time      = time.time()

print("Phase 3 running.")
print("  → Move index finger to control cursor")
print("  → Pinch index + thumb to LEFT CLICK")
print("  → Press 'q' to quit\n")


# ── Helpers ───────────────────────────────────────────────────────────────────

def remap(value, in_min, in_max, out_min, out_max):
    """Clamp + linearly remap value from one range to another."""
    value = max(in_min, min(in_max, value))
    ratio = (value - in_min) / (in_max - in_min)
    return out_min + ratio * (out_max - out_min)


def landmark_distance(lm_a, lm_b):
    """
    Euclidean distance between two MediaPipe landmarks in normalized space.
    Uses only x and y (z is noisy and unnecessary for pinch).
    """
    return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)


def adaptive_alpha(target_x, target_y, prev_x, prev_y):
    """
    Compute EMA alpha based on how fast the finger is moving.
    Fast movement → higher alpha (cursor tracks quickly).
    Still hand → lower alpha (cursor stays rock-solid).
    """
    speed = math.hypot(target_x - prev_x, target_y - prev_y)
    alpha = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * min(1.0, speed * SPEED_SCALE)
    return alpha


def draw_control_zone(frame, w, h):
    x1 = int(CTRL_X_MIN * w);  y1 = int(CTRL_Y_MIN * h)
    x2 = int(CTRL_X_MAX * w);  y2 = int(CTRL_Y_MAX * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 1)
    cv2.putText(frame, "Control Zone", (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1)


def draw_pinch_indicator(frame, lm4, lm8, frame_w, frame_h, is_pinching):
    """Draw a line between thumb and index tip, colour-coded by pinch state."""
    tx = int(lm4.x * frame_w);  ty = int(lm4.y * frame_h)
    ix = int(lm8.x * frame_w);  iy = int(lm8.y * frame_h)
    color = (0, 255, 100) if is_pinching else (180, 180, 180)
    cv2.line(frame, (tx, ty), (ix, iy), color, 2)
    # Midpoint circle shows pinch zone
    mx = (tx + ix) // 2;  my = (ty + iy) // 2
    cv2.circle(frame, (mx, my), 8, color, cv2.FILLED if is_pinching else 1)


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Frame read failed.")
        break

    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    frame_h, frame_w, _ = frame.shape
    now = time.time()

    draw_control_zone(frame, frame_w, frame_h)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw full hand skeleton
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            lm8 = hand_landmarks.landmark[8]   # Index fingertip
            lm4 = hand_landmarks.landmark[4]   # Thumb tip

            # ── Cursor movement ───────────────────────────────────────────
            target_x = remap(lm8.x, CTRL_X_MIN, CTRL_X_MAX, 0, SCREEN_W)
            target_y = remap(lm8.y, CTRL_Y_MIN, CTRL_Y_MAX, 0, SCREEN_H)

            alpha    = adaptive_alpha(target_x, target_y, smooth_x, smooth_y)
            new_x    = alpha * target_x + (1 - alpha) * smooth_x
            new_y    = alpha * target_y + (1 - alpha) * smooth_y

            # Dead zone: only move if change exceeds threshold
            delta = math.hypot(new_x - smooth_x, new_y - smooth_y)
            if delta > DEAD_ZONE_PX:
                smooth_x = new_x
                smooth_y = new_y
                pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # ── Pinch detection ───────────────────────────────────────────
            pinch_dist  = landmark_distance(lm4, lm8)
            is_pinching = pinch_dist < PINCH_THRESHOLD
            cooldown_ok = (now - last_click_t) > CLICK_COOLDOWN_S

            if is_pinching and cooldown_ok:
                pyautogui.click()
                last_click_t  = now
                click_flash_t = now
                print(f"  CLICK at screen ({int(smooth_x)}, {int(smooth_y)})")

            # ── Visual: pinch line + index dot ────────────────────────────
            draw_pinch_indicator(frame, lm4, lm8, frame_w, frame_h, is_pinching)

            ix = int(lm8.x * frame_w)
            iy = int(lm8.y * frame_h)
            cv2.circle(frame, (ix, iy), 10, (0, 0, 255), cv2.FILLED)

            # ── Debug panel ───────────────────────────────────────────────
            debug = [
                f"Pinch dist : {pinch_dist:.3f}  (thresh {PINCH_THRESHOLD})",
                f"Alpha      : {alpha:.3f}",
                f"Cursor     : ({int(smooth_x)}, {int(smooth_y)})",
                f"Dead zone  : {DEAD_ZONE_PX}px | delta {delta:.1f}px",
            ]
            for i, line in enumerate(debug):
                cv2.putText(frame, line,
                            (frame_w - 320, 30 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 50), 1)

            # ── Pinch status label ────────────────────────────────────────
            pinch_label = "PINCHING!" if is_pinching else f"dist: {pinch_dist:.3f}"
            pinch_color = (0, 255, 100) if is_pinching else (160, 160, 160)
            cv2.putText(frame, pinch_label, (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, pinch_color, 2)

        status_text  = "Hand Detected"
        status_color = (0, 220, 0)
    else:
        status_text  = "No hand — show index finger"
        status_color = (0, 100, 255)

    # ── CLICK flash overlay ───────────────────────────────────────────────────
    if (now - click_flash_t) < CLICK_FLASH_S:
        cv2.putText(frame, "CLICK!", (frame_w // 2 - 55, frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 200), 4)

    # ── FPS ───────────────────────────────────────────────────────────────────
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame_h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # ── Status + footer ───────────────────────────────────────────────────────
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "Phase 3: Click Control  |  Pinch = Click  |  Q = quit",
                (20, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    cv2.imshow("Virtual Mouse - Phase 3", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Phase 3.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")