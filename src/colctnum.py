import os
import cv2

cap = cv2.VideoCapture(0)
directory = "Images/"

digit_count = 3000
max_digits = len(str(digit_count - 1))

if not os.path.exists(directory):
    os.makedirs(directory)

while True:
    _, frame = cap.read()
    count = len(os.listdir(directory))

    row = frame.shape[1]
    col = frame.shape[0]
    cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)
    cv2.imshow("data", frame)
    cv2.imshow("ROI", frame[40:400, 0:300])
    frame = frame[40:400, 0:300]
    interrupt = cv2.waitKey(10)

    if interrupt & 0xFF == ord("q") or count >= 10 * digit_count:
        break

    digit = count // digit_count
    filename = f"{digit:01d}/{digit:01d}{count % digit_count + 1:0{max_digits}d}.jpg"
    cv2.imwrite(os.path.join(directory, filename), frame)

cap.release()
cv2.destroyAllWindows()
