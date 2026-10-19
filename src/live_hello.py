

import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque, Counter
from pathlib import Path

MODEL_PATH = Path("models/hello_model.joblib")
SEQUENCE_LENGTH = 60
PREDICTION_THRESHOLD = 3

model = joblib.load(MODEL_PATH)

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

def extract_features(sequence):
    positions = sequence.reshape(60, 225)
    velocity = np.diff(positions, axis=0)

    position_mean = np.mean(positions, axis=0)
    position_std = np.std(positions, axis=0)
    velocity_mean = np.mean(velocity, axis=0)
    velocity_std = np.std(velocity, axis=0)
    velocity_max = np.max(np.abs(velocity), axis=0)

    return np.concatenate([
        position_mean,
        position_std,
        velocity_mean,
        velocity_std,
        velocity_max
    ]).reshape(1, -1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

sequence_buffer = deque(maxlen=SEQUENCE_LENGTH)
prediction_buffer = deque(maxlen=5)

with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)
        image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS
        )

        sequence_buffer.append(extract_landmarks(results))

        label = "COLLECTING FRAMES"

        if len(sequence_buffer) == SEQUENCE_LENGTH:
            sequence = np.asarray(sequence_buffer, dtype=np.float32)
            features = extract_features(sequence)
            predicted = model.predict(features)[0]
            prediction_buffer.append(predicted)

            counts = Counter(prediction_buffer)
            label = counts.most_common(1)[0][0]

            if counts[label] < PREDICTION_THRESHOLD:
                label = "UNCERTAIN"

        color = (0, 255, 0) if label == "HELLO" else (0, 0, 255)

        cv2.putText(
            image, f"Prediction: {label}", (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2
        )
        cv2.putText(
            image, f"Frames: {len(sequence_buffer)}/{SEQUENCE_LENGTH}",
            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
        cv2.putText(
            image, "Press Q to quit", (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
        )

        cv2.imshow("Live HELLO Recognition", image)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

