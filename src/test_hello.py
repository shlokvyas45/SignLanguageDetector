
import cv2
import mediapipe as mp
import numpy as np

DATA_PATH = "data/webcam_asl/HELLO.npy"
SEQUENCE_LENGTH = 60

data = np.load(DATA_PATH)
print("HELLO dataset shape:", data.shape)

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

sequence_index = 0
frame_index = 0

cap = cv2.VideoCapture(0)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

        cv2.putText(
            image, "HELLO DATA: 30 sequences",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        cv2.putText(
            image, "Press Q to quit",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        cv2.imshow("HELLO Sign Test", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
