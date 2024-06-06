from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
import pyautogui
import time


def aslkeyboard():
    sen = "none"
    json_file = open("model.json", "r")
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights("model.h5")

    colors = []
    for i in range(0, 20):
        colors.append((245, 117, 16))
    print(len(colors))

    def prob_viz(res, actio, input_frame, colors, threshold):
        output_frame = input_frame.copy()
        for num, prob in enumerate(res):
            cv2.rectangle(
                output_frame,
                (0, 60 + num * 40),
                (int(prob * 100), 90 + num * 40),
                colors[num],
                -1,
            )
            cv2.putText(
                output_frame,
                actio[num],
                (0, 85 + num * 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return output_frame

    # 1. New detection variables
    sequence = []
    threshold = 0.8

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(
        model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            cropframe = frame[40:400, 0:300]
            # print(frame.shape)
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            # frame=cv2.putText(frame,"Active Region",(75,25),cv2.FONT_HERSHEY_COMPLEX_SMALL,2,255,2)
            image, results = mediapipe_detection(cropframe, hands)
            # print(results)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    for idx, landmark in enumerate(hand_landmarks.landmark):
                        # Convert landmark coordinates to pixel values
                        height, width, _ = frame.shape
                        cx, cy = int(landmark.x * width), int(landmark.y * height)
                        # Draw a circle at each keypoint
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            # Clear sentence and accuracy if no hand landmarks are detected
            if results.multi_hand_landmarks is None:
                sen = "none"

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-1:]

            try:
                if len(sequence) == 1:
                    res = model.predict(np.expand_dims(sequence, axis=0))[0]
                    print([np.argmax(res)])
                    print(actions[np.argmax(res)])
                    sen = actions[np.argmax(res)]

            except Exception as e:
                pass
            if sen == "del":
                sen = "B"
            elif sen == "B":
                sen = "del"

            cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(
                frame,
                f"Output: - {sen}",
                (3, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # If the predicted sentence contains "space", click space
            if sen == "none":
                continue
            elif sen == "space":
                pyautogui.press("space")

            # If the predicted sentence contains "delete", click backspace
            elif sen == "del":
                pyautogui.press("backspace")

            # Otherwise, typewrite the sentence
            else:

                pyautogui.typewrite(sen)
                print("Typed:", sen)

            time.sleep(0.3)

            cv2.imshow("OpenCV Feed", frame)
            cv2.waitKey(3)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


# aslkeyboard()
