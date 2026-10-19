

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "selected_asl" / "selected_asl.npz"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

data = np.load(DATA_PATH)
X = data["X"].astype(np.float32)
y = data["y"]

X = X.reshape(X.shape[0], -1)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", class_weight="balanced"),
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Accuracy:", round(accuracy, 4))
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))

labels = sorted(np.unique(y))
cm = confusion_matrix(y_test, predictions, labels=labels)

print("Labels:", labels)
print("Confusion Matrix:")
print(cm)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels,
)

display.plot(xticks_rotation=45)
plt.tight_layout()
plt.show()

model_path = MODEL_DIR / "asl_model.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path)

