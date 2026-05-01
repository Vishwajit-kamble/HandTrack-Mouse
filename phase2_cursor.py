"""
Virtual Mouse - Phase 2: Coordinate Mapping + Cursor Movement
=============================================================
Goal: Take landmark #8 (index fingertip) from Phase 1,
      map its normalized (0–1) position to screen coordinates,
      and move the actual OS cursor using PyAutoGUI.

New in this phase:
  - Screen resolution detection (works on any monitor size)
  - Coordinate remapping with a "control zone" (avoids edge jitter)
  - Exponential moving average (EMA) smoothing — kills micro-tremor
  - PyAutoGUI cursor movement with failsafe disabled for comfort
  - On-screen debug overlay: raw vs smoothed coords, FPS counter

Install dependencies:
    pip install opencv-python mediapipe pyautogui

Run:
    python phase2_cursor.py

Controls:
    Press 'q' to quit
"""

import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# ── PyAutoGUI config ──────────────────────────────────────────────────────────
pyautogui.FAILSAFE = False   # Prevents crash when cursor hits screen corner
pyautogui.PAUSE = 0          # Remove built-in delay between calls (we handle timing)

SCREEN_W, SCREEN_H = pyautogui.size()   # Auto-detect monitor resolution
print(f"Screen resolution detected: {SCREEN_W} x {SCREEN_H}")

# ── Control zone (the region of the webcam frame that maps to the full screen) ─
# Instead of mapping the ENTIRE frame (0.0–1.0), we use a smaller inner box.
# This makes edge-reaching easier and reduces jitter at extremes.
# Think of it like a mouse pad — you don't need to reach the physical edge.
CTRL_X_MIN = 0.15   # Left boundary  (15% from left)
CTRL_X_MAX = 0.85   # Right boundary (85% from left)
CTRL_Y_MIN = 0.10   # Top boundary   (10% from top)
CTRL_Y_MAX = 0.90   # Bottom boundary(90% from top)

# ── Smoothing config ──────────────────────────────────────────────────────────
# EMA: new_pos = alpha * raw_pos + (1 - alpha) * previous_pos
# Lower alpha → smoother but more lag. Higher → more responsive but jittery.
SMOOTHING_ALPHA = 0.18   # Tuned for natural feel — tweak between 0.1–0.4

# ── MediaPipe setup (same as Phase 1) ────────────────────────────────────────
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

# ── Smoothing state ───────────────────────────────────────────────────────────
# Start at screen center so cursor doesn't snap to (0,0) on first frame
smooth_x = SCREEN_W / 2
smooth_y = SCREEN_H / 2

# ── FPS tracking ──────────────────────────────────────────────────────────────
prev_time = time.time()

print("Phase 2 running. Move your INDEX FINGER to control the cursor.")
print("Press 'q' to quit.\n")


# ── Helper: map a value from one range to another ─────────────────────────────
def remap(value, in_min, in_max, out_min, out_max):
    """
    Like numpy.interp — maps `value` from [in_min, in_max] → [out_min, out_max].
    Clamps output so cursor never flies off-screen.
    """
    value = max(in_min, min(in_max, value))          # clamp input to control zone
    ratio = (value - in_min) / (in_max - in_min)     # normalize to 0.0–1.0
    return out_min + ratio * (out_max - out_min)      # scale to output range


# ── Helper: draw the control zone rectangle on the frame ─────────────────────
def draw_control_zone(frame, w, h):
    x1 = int(CTRL_X_MIN * w)
    y1 = int(CTRL_Y_MIN * h)
    x2 = int(CTRL_X_MAX * w)
    y2 = int(CTRL_Y_MAX * h)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 1)
    cv2.putText(frame, "Control Zone", (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1)


# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Failed to read frame.")
        break

    # --- Mirror + colour convert (same as Phase 1) ---
    frame     = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    frame_h, frame_w, _ = frame.shape

    # --- Always draw the control zone so user knows where to move ---
    draw_control_zone(frame, frame_w, frame_h)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw skeleton (same as Phase 1)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # ── Step A: Get landmark #8 normalized coords ─────────────────
            index_tip = hand_landmarks.landmark[8]
            raw_x = index_tip.x   # 0.0 – 1.0 across webcam frame
            raw_y = index_tip.y   # 0.0 – 1.0 across webcam frame

            # ── Step B: Map camera space → screen space ───────────────────
            # Only the control zone maps to the full screen.
            target_x = remap(raw_x, CTRL_X_MIN, CTRL_X_MAX, 0, SCREEN_W)
            target_y = remap(raw_y, CTRL_Y_MIN, CTRL_Y_MAX, 0, SCREEN_H)

            # ── Step C: EMA smoothing ─────────────────────────────────────
            # Blend new target with previous smoothed position.
            # This removes jitter without adding visible lag.
            smooth_x = SMOOTHING_ALPHA * target_x + (1 - SMOOTHING_ALPHA) * smooth_x
            smooth_y = SMOOTHING_ALPHA * target_y + (1 - SMOOTHING_ALPHA) * smooth_y

            # ── Step D: Move the OS cursor ────────────────────────────────
            pyautogui.moveTo(int(smooth_x), int(smooth_y))

            # ── Overlay: pixel position of fingertip ──────────────────────
            px = int(raw_x * frame_w)
            py = int(raw_y * frame_h)

            # Big red dot on index tip
            cv2.circle(frame, (px, py), 12, (0, 0, 255), cv2.FILLED)
            cv2.putText(frame, "#8", (px + 14, py - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)

            # ── Debug panel (top-right) ───────────────────────────────────
            debug_lines = [
                f"Raw  cam : ({raw_x:.3f}, {raw_y:.3f})",
                f"Target   : ({int(target_x)}, {int(target_y)})",
                f"Smooth   : ({int(smooth_x)}, {int(smooth_y)})",
                f"Screen   : {SCREEN_W}x{SCREEN_H}",
            ]
            for i, line in enumerate(debug_lines):
                cv2.putText(frame, line,
                            (frame_w - 290, 30 + i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 50), 1)

        status_text  = "Hand Detected  |  Moving cursor"
        status_color = (0, 220, 0)

    else:
        # No hand → cursor stays where it is (smooth_x/y unchanged)
        status_text  = "No hand detected — show your index finger"
        status_color = (0, 100, 255)

    # ── FPS counter ───────────────────────────────────────────────────────────
    now      = time.time()
    fps      = 1.0 / (now - prev_time + 1e-9)
    prev_time = now
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, frame_h - 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)

    # ── Status bar ────────────────────────────────────────────────────────────
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame, "Phase 2: Cursor Control  |  Q = quit",
                (20, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

    cv2.imshow("Virtual Mouse - Phase 2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Phase 2.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
hands.close()
print("Done.")