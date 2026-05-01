"""
Virtual Mouse - Phase 1: Webcam + Hand Landmark Detection
=========================================================
Goal: Open webcam, detect hand, draw all 21 landmarks on screen.

Install dependencies first:
    pip install opencv-python mediapipe

Run:
    python phase1_hand_landmarks.py

Controls:
    Press 'q' to quit
"""

import cv2
import mediapipe as mp

# ── MediaPipe setup ───────────────────────────────────────────────────────────
mp_hands   = mp.solutions.hands          # The Hands solution
mp_drawing = mp.solutions.drawing_utils  # Helper to draw landmarks
mp_styles  = mp.solutions.drawing_styles # Predefined nice-looking styles

hands = mp_hands.Hands(
    static_image_mode=False,    # False = video mode (faster, uses tracking)
    max_num_hands=1,            # We only need 1 hand for the mouse
    min_detection_confidence=0.7,  # How confident before declaring a hand found
    min_tracking_confidence=0.5    # How confident to keep tracking (lower = faster)
)

# ── Webcam setup ──────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)       # 0 = default webcam

if not cap.isOpened():
    print("ERROR: Cannot open webcam. Check your camera connection.")
    exit()

# Optional: set resolution (comment out if your webcam doesn't support it)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Phase 1 running. Press 'q' to quit.")
print("Landmark #8 = Index fingertip (highlighted in RED)")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    success, frame = cap.read()     # Read one frame from webcam

    if not success:
        print("ERROR: Failed to read frame from webcam.")
        break

    # --- Step A: Flip the frame horizontally (mirror mode) ---
    # Without this, moving your hand RIGHT moves cursor LEFT. Feels wrong.
    frame = cv2.flip(frame, 1)

    # --- Step B: Convert BGR → RGB ---
    # OpenCV reads frames as BGR. MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- Step C: Process frame through MediaPipe ---
    # We set writeable=False as a performance trick (avoids copying the array)
    rgb_frame.flags.writeable = False
    results = hands.process(rgb_frame)
    rgb_frame.flags.writeable = True

    # --- Step D: Draw landmarks if a hand is detected ---
    frame_h, frame_w, _ = frame.shape   # Height, Width of the frame

    if results.multi_hand_landmarks:
        # There could be multiple hands, but we limited to 1
        for hand_landmarks in results.multi_hand_landmarks:

            # Draw all 21 connections (the "skeleton") + dots
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

            # --- Highlight Landmark #8 (Index Fingertip) in RED ---
            # landmark coordinates are normalized (0.0 to 1.0)
            # multiply by frame size to get pixel coords
            index_tip = hand_landmarks.landmark[8]
            ix = int(index_tip.x * frame_w)   # pixel X
            iy = int(index_tip.y * frame_h)   # pixel Y

            cv2.circle(frame, (ix, iy), 12, (0, 0, 255), cv2.FILLED)  # Big RED dot
            cv2.putText(
                frame, "#8 Index Tip",
                (ix + 15, iy),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
            )

            # --- Print coordinates to console (useful for debugging) ---
            print(f"  Landmark #8 → pixel ({ix}, {iy})  |  normalized ({index_tip.x:.3f}, {index_tip.y:.3f})")

        status_text = "Hand Detected!"
        status_color = (0, 200, 0)   # Green
    else:
        status_text = "No hand detected — show your hand"
        status_color = (0, 100, 255) # Orange

    # --- Step E: Draw status overlay on the frame ---
    cv2.putText(frame, status_text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

    cv2.putText(frame, "Phase 1: Landmark Detection | Q = quit", (20, frame_h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # --- Step F: Show the frame ---
    cv2.imshow("Virtual Mouse - Phase 1", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Quitting Phase 1.")
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()           # Release webcam
cv2.destroyAllWindows() # Close all OpenCV windows
hands.close()           # Free MediaPipe resources
print("Done.")