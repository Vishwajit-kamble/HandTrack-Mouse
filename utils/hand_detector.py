"""
utils/hand_detector.py
======================
MediaPipe Hands wrapper.
Tracks: index tip, thumb tip, middle tip, ring tip
Computes: pinch distances for left-click, right-click, scroll, and volume.
"""

import cv2
import mediapipe as mp
import math


class HandDetector:
    # Fixed MediaPipe landmark indices
    WRIST      = 0
    THUMB_TIP  = 4
    INDEX_TIP  = 8
    MIDDLE_TIP = 12
    RING_TIP   = 16
    PINKY_TIP  = 20

    def __init__(self, max_hands=1, detection_confidence=0.75, tracking_confidence=0.6):
        self._mp_hands   = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles  = mp.solutions.drawing_styles

        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )

        # ── Public state (updated every process_frame call) ───────────────────
        self.hand_detected       = False
        self.landmarks           = None

        self.index_tip           = None   # lm #8
        self.thumb_tip           = None   # lm #4
        self.middle_tip          = None   # lm #12
        self.ring_tip            = None   # lm #16

        self.pinch_distance      = 1.0   # thumb ↔ index  → left click / volume
        self.rclick_distance     = 1.0   # thumb ↔ middle → right click
        self.scroll_distance     = 1.0   # index ↔ middle → scroll mode trigger
        self.prev_pinch_distance = 1.0   # previous frame (for volume delta)

        # Y position of index tip — used to compute scroll direction/speed
        self.index_y_norm        = 0.5   # normalized 0–1
        self.prev_index_y_norm   = 0.5

    # ── Core ──────────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """Run MediaPipe on one BGR frame. Updates all public attributes."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)
        rgb.flags.writeable = True

        # Save previous values before overwriting
        self.prev_pinch_distance = self.pinch_distance
        self.prev_index_y_norm   = self.index_y_norm

        if results.multi_hand_landmarks:
            self.hand_detected  = True
            self.landmarks      = results.multi_hand_landmarks[0]
            lm = self.landmarks.landmark

            self.index_tip      = lm[self.INDEX_TIP]
            self.thumb_tip      = lm[self.THUMB_TIP]
            self.middle_tip     = lm[self.MIDDLE_TIP]
            self.ring_tip       = lm[self.RING_TIP]

            self.pinch_distance  = self._dist(self.thumb_tip,  self.index_tip)
            self.rclick_distance = self._dist(self.thumb_tip,  self.middle_tip)
            self.scroll_distance = self._dist(self.index_tip,  self.middle_tip)

            self.index_y_norm    = self.index_tip.y   # 0 = top, 1 = bottom
        else:
            self.hand_detected   = False
            self.landmarks       = None
            self.index_tip       = self.thumb_tip = self.middle_tip = self.ring_tip = None
            self.pinch_distance  = 1.0
            self.rclick_distance = 1.0
            self.scroll_distance = 1.0

        return results

    # ── Drawing ───────────────────────────────────────────────────────────────

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
            cv2.circle(frame,
                       (int(lm.x * w), int(lm.y * h)),
                       radius, color, cv2.FILLED)

    def draw_gesture_line(self, frame, lm_a, lm_b, threshold, label="", active_color=(0,255,100)):
        """Line between two landmarks — changes colour when gesture is active."""
        if lm_a and lm_b:
            h, w, _ = frame.shape
            ax, ay = int(lm_a.x * w), int(lm_a.y * h)
            bx, by = int(lm_b.x * w), int(lm_b.y * h)
            dist   = self._dist(lm_a, lm_b)
            active = dist < threshold
            color  = active_color if active else (150, 150, 150)
            cv2.line(frame, (ax, ay), (bx, by), color, 2)
            mx, my = (ax + bx) // 2, (ay + by) // 2
            cv2.circle(frame, (mx, my), 7, color, cv2.FILLED if active else 1)
            if label:
                cv2.putText(frame, label, (mx + 9, my - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    def draw_scroll_indicator(self, frame, lm_a, lm_b, threshold, is_scrolling, direction):
        """
        Draw a special scroll indicator: circle around both fingertips
        + animated arrow showing scroll direction.
        """
        if not (lm_a and lm_b):
            return
        h, w, _ = frame.shape
        ax, ay = int(lm_a.x * w), int(lm_a.y * h)
        bx, by = int(lm_b.x * w), int(lm_b.y * h)
        dist   = self._dist(lm_a, lm_b)
        active = dist < threshold

        # Line between the two tips
        color = (80, 200, 255) if active else (150, 150, 150)
        cv2.line(frame, (ax, ay), (bx, by), color, 2)

        # Circle enclosing both tips (scroll zone indicator)
        if active:
            mx, my = (ax + bx) // 2, (ay + by) // 2
            radius = int(self._dist(lm_a, lm_b) * w / 2) + 20
            cv2.circle(frame, (mx, my), max(radius, 22), (80, 200, 255), 2)

            # Arrow showing scroll direction
            if direction != 0:
                arrow_len = 35
                arrow_tip = (mx, my - arrow_len) if direction < 0 else (mx, my + arrow_len)
                cv2.arrowedLine(frame, (mx, my), arrow_tip,
                                (80, 200, 255), 2, tipLength=0.4)
                label = "SCROLL UP" if direction < 0 else "SCROLL DOWN"
                cv2.putText(frame, label, (mx - 50, my - arrow_len - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255), 2)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _dist(a, b):
        return math.hypot(a.x - b.x, a.y - b.y)

    def close(self):
        self._hands.close()