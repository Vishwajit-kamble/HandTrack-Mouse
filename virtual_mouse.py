import cv2
from utils.hand_detector import HandDetector
from utils.mouse_controller import MouseController

cam = cv2.VideoCapture(0)

detector = HandDetector()
mouse = MouseController(smoothening=5)

while True:
    success, frame = cam.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    frame = detector.find_hands(frame)
    lm_list = detector.get_landmarks(frame)

    if lm_list:
        ix, iy = lm_list[8][1], lm_list[8][2]
        tx, ty = lm_list[4][1], lm_list[4][2]

        mouse.move(ix, iy, w, h)

        if mouse.click(ix, iy, tx, ty):
            cv2.circle(frame, (ix, iy), 15, (0, 0, 255), cv2.FILLED)

        cv2.circle(frame, (ix, iy), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (tx, ty), 10, (255, 0, 0), cv2.FILLED)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()