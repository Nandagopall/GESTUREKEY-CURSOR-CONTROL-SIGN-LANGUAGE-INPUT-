import cv2
import mediapipe as mp
from screeninfo import get_monitors
from pynput.mouse import Button, Controller
import numpy as np


def hand_mouse_control():
    capture = cv2.VideoCapture(0)

    m_draw = mp.solutions.drawing_utils
    m_hand = mp.solutions.hands
    hands = m_hand.Hands(max_num_hands=1)

    inc = 6

    prex, prey, curx, cury = 0, 0, 0, 0
    lclicked, rclicked = False, False
    thumb_down_count = 0
    scroll_amount = 0  # Variable to control scrolling

    mou = Controller()

    def get_st(pos):
        # up or down
        st = [False] * 5
        index = ((6, 8), (10, 12), (14, 16), (18, 20), (2, 4))

        for i, f in enumerate(index):
            if pos[f[0]].y > pos[f[1]].y:
                st[i] = True

        return st

    def mov_mou(pos, image_w, image_h):
        # move cursor
        nonlocal prex, prey, curx, cury

        monitor = get_monitors()[0]

        m_w = monitor.width
        m_h = monitor.height

        x_pos = sum([pos[i].x for i in range(5, 18, 4)]) / 4
        y_pos = sum([pos[i].y for i in range(5, 18, 4)]) / 4

        pos = (x_pos * image_w, y_pos * image_h)

        x = np.interp(pos[0], (200, image_w - 200), (0, m_w))
        y = np.interp(pos[1], (200, image_h - 200), (0, m_h))

        curx = prex + (x - prex) / inc
        cury = prey + (y - prey) / inc

        mou.position = (int(m_w - curx), int(cury))
        prex, prey = curx, cury

    def cl(st):
        # handle button clicks
        nonlocal lclicked, rclicked, thumb_down_count, scroll_amount
        if not st[0] and st[1] and not lclicked:
            mou.press(Button.left)
            mou.release(Button.left)
            lclicked = True

        if st[0] and not st[1] and not rclicked:
            mou.press(Button.right)
            mou.release(Button.right)
            rclicked = True

        if st[2] and not st[3]:  # Ring finger down for scroll up
            scroll_amount = -1  # Scroll up
        elif st[3] and not st[2] and st[0]:  # Little finger down for scroll down
            scroll_amount = 1  # Scroll down
        else:
            scroll_amount = 0  # Stop scrolling

        # Perform scrolling
        mou.scroll(0, scroll_amount)
        if not st[3] and not st[2] and not st[1] and not st[0]:  # down  double click

            mou.click(Button.left, 2)  # Perform double click

        if st[0] and st[1]:
            lclicked = False
            rclicked = False
        if not st[2] and not st[3]:
            scroll_amount = 0

    while True:
        success, img = capture.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rslts = hands.process(imgRGB)
        h, w, temp = img.shape

        if rslts.multi_hand_landmarks:
            for landmarks in rslts.multi_hand_landmarks:
                up_fingers = get_st(landmarks.landmark)
                cl(up_fingers)
                mov_mou(landmarks.landmark, w, h)
                m_draw.draw_landmarks(img, landmarks, m_hand.HAND_CONNECTIONS)

        cv2.rectangle(img, (200, 200), (w - 200, h - 200), (0, 255, 0), 3)
        cv2.imshow("Real Time", cv2.flip(img, 1))
        if cv2.waitKey(1) & 0xFF == 27:
            break
        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


# Call the function to run the hand mouse control
# hand_mouse_control()
