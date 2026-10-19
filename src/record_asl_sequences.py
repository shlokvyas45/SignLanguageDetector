

import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path

SAVE_DIR = Path("data/webcam_asl")
LABEL = "NO_SIGN"
SEQUENCES = 30
FRAMES = 60

SAVE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = SAVE_DIR / f"{LABEL}.npy"

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):
    pose = np.zeros((33, 3), dtype=np.float32)
    left_hand = np.zeros((21, 3), dtype=np.float32)
    right_hand = np.zeros((21, 3), dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array(
            [[p.x, p.y, p.z] for p in results.pose_landmarks.landmark],
            dtype=np.float32
        )

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[p.x, p.y, p.z] for p in results.left_hand_landmarks.landmark],
            dtype=np.float32
        )

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[p.x, p.y, p.z] for p in results.right_hand_landmarks.landmark],
            dtype=np.float32
        )

    return np.concatenate([pose, left_hand, right_hand], axis=0)

cap = cv2.VideoCapture(0)
recorded = []

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened() and len(recorded) < SEQUENCES:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
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
            image, f"{LABEL}: {len(recorded)}/{SEQUENCES}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )
        cv2.putText(
            image, "Stay still naturally | S: record | Q: save and quit",
            (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )

        cv2.imshow("ASL Dataset Recorder", image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s"):
            sequence = []

            for _ in range(FRAMES):
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image_rgb)
                sequence.append(extract_landmarks(results))

                display = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    display, f"Recording {len(sequence)}/{FRAMES}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2
                )
                cv2.imshow("ASL Dataset Recorder", display)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if len(sequence) == FRAMES:
                recorded.append(np.array(sequence, dtype=np.float32))
                print(f"Recorded {len(recorded)}/{SEQUENCES}")
            else:
                print("Incomplete sequence discarded.")

        elif key == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

if recorded:
    new_data = np.array(recorded, dtype=np.float32)

    if SAVE_PATH.exists():
        old_data = np.load(SAVE_PATH)
        new_data = np.concatenate([old_data, new_data], axis=0)

    np.save(SAVE_PATH, new_data)
    print("Saved:", SAVE_PATH)
    print("Final shape:", new_data.shape)
else:
    print("No sequences recorded.")
