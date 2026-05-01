import pyautogui
import numpy as np
import math
import time

class MouseController:
    def __init__(self, smoothening=5):
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = 0, 0
        self.smoothening = smoothening
        self.last_click_time = 0

    def move(self, x, y, frame_w, frame_h):
        screen_x = np.interp(x, [0, frame_w], [0, self.screen_w])
        screen_y = np.interp(y, [0, frame_h], [0, self.screen_h])

        curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening

        pyautogui.moveTo(curr_x, curr_y)

        self.prev_x, self.prev_y = curr_x, curr_y

    def click(self, x1, y1, x2, y2, threshold=30):
        distance = math.hypot(x2 - x1, y2 - y1)

        if distance < threshold:
            current_time = time.time()
            if current_time - self.last_click_time > 0.3:
                pyautogui.click()
                self.last_click_time = current_time
                return True

        return False