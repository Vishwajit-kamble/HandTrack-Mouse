"""
utils/hand_detector.py
======================
MediaPipe Hands wrapper.
Handles all setup, processing, and landmark extraction in one clean class.
Import this anywhere you need hand detection — no MediaPipe boilerplate needed.
"""

import cv2
import mediapipe as mp
import math


class HandDetector:
    """
    Wraps MediaPipe Hands. Call process_frame() each loop iteration,
    then read .index_tip, .thumb_tip, .pinch_distance, etc.
    """

    # MediaPipe landmark indices (fixed by the model — never change)
    WRIST         = 0
    THUMB_TIP     = 4
    INDEX_TIP     = 8
    MIDDLE_TIP    = 12
    RING_TIP      = 16
    PINKY_TIP     = 20

    def __init__(
        self,
        max_hands=1,
        detection_confidence=0.7,
        tracking_confidence=0.5,
    ):
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # Public state — updated every call to process_frame()
        self.hand_detected  = False
        self.landmarks      = None   # raw MediaPipe landmark object
        self.index_tip      = None   # landmark #8  (x, y normalized)
        self.thumb_tip      = None   # landmark #4  (x, y normalized)
        self.pinch_distance = 1.0    # distance between #4 and #8

    # ── Core method ───────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Run MediaPipe on one BGR frame. Updates all public attributes.
        Returns the same frame (unmodified — drawing is separate).
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        if results.multi_hand_landmarks:
            self.hand_detected = True
            self.landmarks     = results.multi_hand_landmarks[0]  # first hand only
            self.index_tip     = self.landmarks.landmark[self.INDEX_TIP]
            self.thumb_tip     = self.landmarks.landmark[self.THUMB_TIP]
            self.pinch_distance = self._distance(self.thumb_tip, self.index_tip)
        else:
            self.hand_detected  = False
            self.landmarks      = None
            self.index_tip      = None
            self.thumb_tip      = None
            self.pinch_distance = 1.0

        return results

    # ── Drawing helpers ───────────────────────────────────────────────────────

    def draw_landmarks(self, frame):
        """Draw the full hand skeleton onto the frame (in-place)."""
        if self.landmarks:
            self._mp_drawing.draw_landmarks(
                frame,
                self.landmarks,
                self._mp_hands.HAND_CONNECTIONS,
                self._mp_styles.get_default_hand_landmarks_style(),
                self._mp_styles.get_default_hand_connections_style(),
            )

    def draw_index_dot(self, frame, radius=10, color=(0, 0, 255)):
        """Draw a filled circle on the index fingertip."""
        if self.index_tip:
            h, w, _ = frame.shape
            cx = int(self.index_tip.x * w)
            cy = int(self.index_tip.y * h)
            cv2.circle(frame, (cx, cy), radius, color, cv2.FILLED)

    def draw_pinch_line(self, frame, threshold):
        """
        Draw a line between thumb and index.
        Turns green when pinching, grey otherwise.
        """
        if self.index_tip and self.thumb_tip:
            h, w, _ = frame.shape
            tx = int(self.thumb_tip.x * w);  ty = int(self.thumb_tip.y * h)
            ix = int(self.index_tip.x * w);  iy = int(self.index_tip.y * h)
            color = (0, 255, 100) if self.pinch_distance < threshold else (180, 180, 180)
            cv2.line(frame, (tx, ty), (ix, iy), color, 2)
            mx = (tx + ix) // 2;  my = (ty + iy) // 2
            cv2.circle(frame, (mx, my), 7, color,
                       cv2.FILLED if self.pinch_distance < threshold else 1)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _distance(lm_a, lm_b):
        """Euclidean distance between two landmarks in normalized (x,y) space."""
        return math.hypot(lm_a.x - lm_b.x, lm_a.y - lm_b.y)

    def close(self):
        """Release MediaPipe resources."""
        self._hands.close()