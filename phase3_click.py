import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

cam = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(False, 1, 0.7, 0.7)
draw = mp.solutions.drawing_utils

prev_x, prev_y = 0, 0
smoothening = 5

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand in result.multi_hand_landmarks:
            draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark

            ix, iy = int(lm[8].x * w), int(lm[8].y * h)
            tx, ty = int(lm[4].x * w), int(lm[4].y * h)

            screen_x = np.interp(lm[8].x, [0, 1], [0, screen_w])
            screen_y = np.interp(lm[8].y, [0, 1], [0, screen_h])

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
            cv2.circle(frame, (tx, ty), 10, (255, 0, 0), cv2.FILLED)

            distance = math.hypot(tx - ix, ty - iy)

            if distance < 30:
                pyautogui.click()
                cv2.circle(frame, (ix, iy), 15, (0, 0, 255), cv2.FILLED)

    cv2.imshow("Virtual Mouse - Phase 3", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()