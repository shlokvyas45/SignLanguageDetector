

import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

DATA_DIR = Path("data/webcam_asl")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

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
    ])

X = []
y = []

for label in ["HELLO", "NO_SIGN"]:
    path = DATA_DIR / f"{label}.npy"

    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    sequences = np.load(path)

    if sequences.shape[1:] != (60, 75, 3):
        raise ValueError(f"Unexpected shape for {label}: {sequences.shape}")

    for sequence in sequences:
        X.append(extract_features(sequence))
        y.append(label)

X = np.asarray(X, dtype=np.float32)
y = np.asarray(y)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, class_weight="balanced")
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Test accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\n", classification_report(y_test, predictions, zero_division=0))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions, labels=["HELLO", "NO_SIGN"]))

joblib.dump(model, MODEL_DIR / "hello_model.joblib")
print("Updated model saved.")

