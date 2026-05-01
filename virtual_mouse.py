"""
Virtual Mouse — Final App
=========================
All 3 phases combined in one clean file, powered by utils/.

    Phase 1 → webcam + landmark detection   (HandDetector)
    Phase 2 → coordinate mapping + movement (MouseController)
    Phase 3 → smoothing + pinch-to-click    (MouseController)

Project structure:
    virtual_mouse/
    ├── virtual_mouse.py        ← YOU ARE HERE (run this)
    ├── requirements.txt
    └── utils/
        ├── __init__.py
        ├── hand_detector.py
        └── mouse_controller.py

Install:
    pip install opencv-python mediapipe pyautogui

Run:
    python virtual_mouse.py

Controls:
    Pinch index finger + thumb  →  LEFT CLICK
    Press 'q'                   →  Quit
    Press 'd'                   →  Toggle debug overlay
    Press 'c'                   →  Toggle control zone overlay
"""

import cv2
import time
import sys
import os

# ── Make sure utils/ is importable regardless of working directory ────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.hand_detector   import HandDetector
from utils.mouse_controller import MouseController

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG  — tweak these to tune behaviour
# ═════════════════════════════════════════════════════════════════════════════

WEBCAM_INDEX    = 0      # 0 = default camera; try 1 or 2 if wrong camera opens
FRAME_WIDTH     = 640
FRAME_HEIGHT    = 480

# Control zone (fraction of frame that maps to full screen)
CTRL_X_MIN, CTRL_X_MAX = 0.15, 0.85
CTRL_Y_MIN, CTRL_Y_MAX = 0.10, 0.90

# Smoothing (adaptive EMA)
ALPHA_MIN   = 0.10   # steadiness at rest
ALPHA_MAX   = 0.45   # responsiveness during fast swipes
SPEED_SCALE = 0.008  # how fast alpha ramps up

DEAD_ZONE_PX = 3     # sub-pixel tremor filter

# Pinch-to-click
PINCH_THRESHOLD  = 0.052   # normalized distance; lower = harder to trigger
CLICK_COOLDOWN   = 0.5     # seconds between clicks
CLICK_FLASH_DUR  = 0.25    # seconds the "CLICK!" text shows

# ═════════════════════════════════════════════════════════════════════════════
#  INIT
# ═════════════════════════════════════════════════════════════════════════════

detector = HandDetector(
    max_hands=1,
    detection_confidence=0.7,
    tracking_confidence=0.5,
)

controller = MouseController(
    ctrl_x_min=CTRL_X_MIN, ctrl_x_max=CTRL_X_MAX,
    ctrl_y_min=CTRL_Y_MIN, ctrl_y_max=CTRL_Y_MAX,
    alpha_min=ALPHA_MIN,   alpha_max=ALPHA_MAX,
    speed_scale=SPEED_SCALE,
    dead_zone_px=DEAD_ZONE_PX,
    pinch_threshold=PINCH_THRESHOLD,
    click_cooldown=CLICK_COOLDOWN,
    click_flash_duration=CLICK_FLASH_DUR,
)

cap = cv2.VideoCapture(WEBCAM_INDEX)
if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check WEBCAM_INDEX in config.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

# UI toggles
show_debug        = True
show_control_zone = True
prev_time         = time.time()

print("=" * 52)
print("  Virtual Mouse — running")
print(f"  Screen : {controller.screen_w} x {controller.screen_h}")
print(f"  Camera : {FRAME_WIDTH}x{FRAME_HEIGHT}  index={WEBCAM_INDEX}")
print("  Pinch index + thumb to click")
print("  Q = quit  |  D = debug  |  C = control zone")
print("=" * 52)


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def draw_control_zone(frame, fw, fh):
    x1 = int(CTRL_X_MIN * fw);  y1 = int(CTRL_Y_MIN * fh)
    x2 = int(CTRL_X_MAX * fw);  y2 = int(CTRL_Y_MAX * fh)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 255), 1)
    cv2.putText(frame, "Control Zone", (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 80, 255), 1)


def draw_debug_panel(frame, fw, detector, controller, fps, pinch_dist):
    cx, cy = controller.cursor_pos
    lines = [
        f"FPS        : {fps:.1f}",
        f"Cursor     : ({cx}, {cy})",
        f"Pinch dist : {pinch_dist:.3f}  (thr {PINCH_THRESHOLD})",
        f"Dead zone  : {DEAD_ZONE_PX}px",
        f"Alpha range: {ALPHA_MIN}–{ALPHA_MAX}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (fw - 310, 28 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 50), 1)


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═════════════════════════════════════════════════════════════════════════════

while True:
    success, frame = cap.read()
    if not success:
        print("ERROR: Frame read failed.")
        break

    # ── Pre-process ───────────────────────────────────────────────────────────
    frame = cv2.flip(frame, 1)                  # mirror (feels natural)
    fh, fw, _ = frame.shape

    # ── Hand detection ────────────────────────────────────────────────────────
    detector.process_frame(frame)               # updates detector.* attributes

    # ── Overlays: control zone ────────────────────────────────────────────────
    if show_control_zone:
        draw_control_zone(frame, fw, fh)

    # ── Hand logic ────────────────────────────────────────────────────────────
    if detector.hand_detected:

        # Draw skeleton + index dot + pinch line
        detector.draw_landmarks(frame)
        detector.draw_index_dot(frame, radius=10, color=(0, 0, 255))
        detector.draw_pinch_line(frame, PINCH_THRESHOLD)

        # Move cursor using index fingertip
        controller.move(detector.index_tip.x, detector.index_tip.y)

        # Attempt click on pinch
        clicked = controller.try_click(detector.pinch_distance)
        if clicked:
            cx, cy = controller.cursor_pos
            print(f"  CLICK at screen ({cx}, {cy})")

        # Pinch status label
        is_pinching  = detector.pinch_distance < PINCH_THRESHOLD
        pinch_label  = "PINCHING!" if is_pinching else f"dist: {detector.pinch_distance:.3f}"
        pinch_color  = (0, 255, 100) if is_pinching else (160, 160, 160)
        cv2.putText(frame, pinch_label, (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, pinch_color, 2)

        status_text  = "Hand Detected  |  Move index finger"
        status_color = (0, 220, 0)

    else:
        status_text  = "No hand — show your index finger"
        status_color = (0, 100, 255)

    # ── CLICK flash ───────────────────────────────────────────────────────────
    if controller.click_flash_active:
        cv2.putText(frame, "CLICK!", (fw // 2 - 70, fh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 255, 180), 4)

    # ── FPS ───────────────────────────────────────────────────────────────────
    now       = time.time()
    fps       = 1.0 / (now - prev_time + 1e-9)
    prev_time = now

    # ── Debug panel ───────────────────────────────────────────────────────────
    if show_debug:
        draw_debug_panel(frame, fw, detector, controller, fps,
                         detector.pinch_distance)

    # ── Status bar + footer ───────────────────────────────────────────────────
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.putText(frame,
                "Pinch=Click | Q=Quit | D=Debug | C=CtrlZone",
                (20, fh - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    # ── Show frame ────────────────────────────────────────────────────────────
    cv2.imshow("Virtual Mouse", frame)

    # ── Key handling ──────────────────────────────────────────────────────────
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("Quitting.")
        break
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"  Debug overlay: {'ON' if show_debug else 'OFF'}")
    elif key == ord('c'):
        show_control_zone = not show_control_zone
        print(f"  Control zone: {'ON' if show_control_zone else 'OFF'}")

# ═════════════════════════════════════════════════════════════════════════════
#  CLEANUP
# ═════════════════════════════════════════════════════════════════════════════

cap.release()
cv2.destroyAllWindows()
detector.close()
print("Done.")