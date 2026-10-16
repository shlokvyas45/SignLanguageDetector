
import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

LABEL = input("Enter dataset label: ").strip().upper()

if not LABEL or not LABEL.replace("_", "").isalnum():
    print("Invalid label!")
    exit()
SAMPLES = 100

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera open nahi ho raha hai!")
    exit()

recorded_samples = []

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    print("Hand camera ke saamne rakho.")
    print("Recording start karne ke liye S dabao.")
    print("Exit karne ke liye Q dabao.")

    recording = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Camera frame nahi mil raha!")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS
            )

            landmarks = np.array([
                [point.x, point.y, point.z]
                for point in hand.landmark
            ], dtype=np.float32)

            if recording:
                recorded_samples.append(landmarks)
                cv2.putText(
                    frame,
                    f"Recording: {len(recorded_samples)}/{SAMPLES}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                if len(recorded_samples) >= SAMPLES:
                    recording = False
                    np.save(
                        DATA_DIR / f"{LABEL}.npy",
                        np.array(recorded_samples, dtype=np.float32)
                    )
                    print(f"Saved {len(recorded_samples)} samples.")
                    break
        else:
            cv2.putText(
                frame,
                "No hand detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        cv2.imshow("ISL Dataset Recorder", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            recording = True
        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
