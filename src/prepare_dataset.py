
import numpy as np
from pathlib import Path

DATA_DIR = Path("data")

X = []
y = []

for file in DATA_DIR.glob("*.npy"):
    if file.stem == "TEST":
        continue

    samples = np.load(file)

    if samples.ndim != 3 or samples.shape[1:] != (21, 3):
        print(f"Skipping invalid file: {file.name}")
        continue

    label = file.stem

    for sample in samples:
        X.append(sample)
        y.append(label)

if not X:
    print("No valid labeled data found.")
    exit()

X = np.array(X, dtype=np.float32)
y = np.array(y)

np.savez_compressed(
    DATA_DIR / "training_dataset.npz",
    X=X,
    y=y
)

print("Dataset prepared successfully!")
print("Features shape:", X.shape)
print("Labels shape:", y.shape)
print("Labels:", np.unique(y))
