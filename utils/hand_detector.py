"""
utils/hand_detector.py
======================
MediaPipe Hands wrapper — supports index, thumb, and middle fingertip
for left-click, right-click, and volume gestures.
"""

import cv2
import mediapipe as mp
import math


class HandDetector:
    WRIST      = 0
    THUMB_TIP  = 4
    INDEX_TIP  = 8
    MIDDLE_TIP = 12
    RING_TIP   = 16
    PINKY_TIP  = 20

    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.5):
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        self.hand_detected       = False
        self.landmarks           = None
        self.index_tip           = None
        self.thumb_tip           = None
        self.middle_tip          = None
        self.pinch_distance      = 1.0   # thumb ↔ index  → left click / volume
        self.rclick_distance     = 1.0   # thumb ↔ middle → right click
        self.prev_pinch_distance = 1.0   # previous frame pinch (for volume delta)

    def process_frame(self, frame):
        """Run detection on one BGR frame. Updates all public attributes."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        self.prev_pinch_distance = self.pinch_distance   # save before overwriting

        if results.multi_hand_landmarks:
            self.hand_detected   = True
            self.landmarks       = results.multi_hand_landmarks[0]
            lm                   = self.landmarks.landmark
            self.index_tip       = lm[self.INDEX_TIP]
            self.thumb_tip       = lm[self.THUMB_TIP]
            self.middle_tip      = lm[self.MIDDLE_TIP]
            self.pinch_distance  = self._dist(self.thumb_tip, self.index_tip)
            self.rclick_distance = self._dist(self.thumb_tip, self.middle_tip)
        else:
            self.hand_detected   = False
            self.landmarks       = None
            self.index_tip       = self.thumb_tip = self.middle_tip = None
            self.pinch_distance  = 1.0
            self.rclick_distance = 1.0

        return results

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def draw_landmarks(self, frame):
        if self.landmarks:
            self._mp_drawing.draw_landmarks(
                frame, self.landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

    def draw_tip_dot(self, frame, lm, radius=10, color=(0, 0, 255)):
        if lm:
            h, w, _ = frame.shape
            cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), radius, color, cv2.FILLED)

    def draw_gesture_line(self, frame, lm_a, lm_b, threshold, label=""):
        """Draw line between two landmarks. Green = active gesture."""
        if lm_a and lm_b:
            h, w, _ = frame.shape
            ax, ay = int(lm_a.x * w), int(lm_a.y * h)
            bx, by = int(lm_b.x * w), int(lm_b.y * h)
            dist   = self._dist(lm_a, lm_b)
            color  = (0, 255, 100) if dist < threshold else (150, 150, 150)
            cv2.line(frame, (ax, ay), (bx, by), color, 2)
            mx, my = (ax + bx) // 2, (ay + by) // 2
            cv2.circle(frame, (mx, my), 7, color,
                       cv2.FILLED if dist < threshold else 1)
            if label:
                cv2.putText(frame, label, (mx + 9, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def close(self):
        self._hands.close()