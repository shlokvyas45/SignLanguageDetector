

from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

train_data = np.load(ROOT / "data" / "selected_asl" / "selected_asl.npz")
val_data = np.load(ROOT / "data" / "selected_asl" / "validation_asl.npz")

X_train = train_data["X"].astype(np.float32).reshape(len(train_data["y"]), -1)
y_train = train_data["y"]

X_val = val_data["X"].astype(np.float32).reshape(len(val_data["y"]), -1)
y_val = val_data["y"]

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000, class_weight="balanced"),
)

model.fit(X_train, y_train)
predictions = model.predict(X_val)

print("Validation samples:", len(y_val))
print("Validation accuracy:", round(accuracy_score(y_val, predictions), 4))
print("\nClassification Report:")
print(classification_report(y_val, predictions, zero_division=0))

labels = sorted(np.unique(y_train))
print("Label order:", labels)
print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions, labels=labels))
