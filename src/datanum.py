from function import *
from time import sleep

no_sequences = 3000

sequence_length = 1
for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

no_sequences = 1500

with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:

    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                # ret, frame = cap.read()
                image_path = "Images/{}/{}_{}.jpg".format(action, action, sequence + 1)
                print(image_path)
                # print(f"READ{sequence}")
                frame = cv2.imread(image_path)
                if frame is None:
                    # print("none")
                    continue  # Skip if image not found

                # Make detections
                image, results = mediapipe_detection(frame, hands)
                #                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)
                print("land")

                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(
                        image,
                        "STARTING COLLECTION",
                        (120, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        4,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Show to screen
                    cv2.imshow("OpenCV Feed", image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(
                        image,
                        "Collecting frames for {} Video Number {}".format(
                            action, sequence
                        ),
                        (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                    # Show to screen
                    cv2.imshow("OpenCV Feed", image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(
                    DATA_PATH, action, str(sequence), str(frame_num)
                )
                np.save(npy_path, keypoints)
                print("Saved")

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break

    # cap.release()
    cv2.destroyAllWindows()
