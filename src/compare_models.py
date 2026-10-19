
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

ROOT = Path(__file__).resolve().parent.parent
data = np.load(ROOT / "data" / "selected_asl" / "selected_asl.npz")

X = data["X"].astype(np.float32).reshape(len(data["y"]), -1)
y = data["y"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "SVM RBF C=0.1": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=0.1, class_weight="balanced"),
    ),
    "SVM RBF C=1": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1, class_weight="balanced"),
    ),
    "SVM RBF C=10": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, class_weight="balanced"),
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        ),
    ),
    "KNN k=3": make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=3),
    ),
}

print("5-fold cross-validation on training data")
print("Total samples:", len(y))
print()

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    print(
        f"{name}: mean={scores.mean():.4f}, "
        f"std={scores.std():.4f}, folds={np.round(scores, 3)}"
    )
