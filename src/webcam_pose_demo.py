
import cv2
import mediapipe as mp
import numpy as np

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
) as holistic:
    while cap.isOpened():
        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        pose = np.zeros((33, 3), dtype=np.float32)
        left_hand = np.zeros((21, 3), dtype=np.float32)
        right_hand = np.zeros((21, 3), dtype=np.float32)

        if results.pose_landmarks:
            pose = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.pose_landmarks.landmark
            ], dtype=np.float32)
            mp_draw.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_holistic.POSE_CONNECTIONS
            )

        if results.left_hand_landmarks:
            left_hand = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.left_hand_landmarks.landmark
            ], dtype=np.float32)
            mp_draw.draw_landmarks(
                frame,
                results.left_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        if results.right_hand_landmarks:
            right_hand = np.array([
                [lm.x, lm.y, lm.z]
                for lm in results.right_hand_landmarks.landmark
            ], dtype=np.float32)
            mp_draw.draw_landmarks(
                frame,
                results.right_hand_landmarks,
                mp_holistic.HAND_CONNECTIONS
            )

        landmarks = np.concatenate([pose, left_hand, right_hand])
        cv2.putText(
            frame,
            f"Landmarks: {landmarks.shape[0]} x {landmarks.shape[1]}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("ASL 75-Landmark Test - Press Q to Quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
