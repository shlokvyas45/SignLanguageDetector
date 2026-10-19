
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "asl_citizen"
OUTPUT_DIR = ROOT / "data" / "selected_asl"
BASE_URL = (
    "https://huggingface.co/datasets/"
    "SharoonArshad/asl-citizen-processed-200/resolve/main/"
)

SELECTED_LABELS = {"HELLO", "MORNING", "NAME", "TWO", "NIGHT1"}
FEATURE_BYTES = 200 * 450 * 2
DATA_OFFSET = 128

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(DATA_DIR / "metadata" / "val_manifest.csv")
manifest = manifest[manifest["label"].isin(SELECTED_LABELS)]

samples = []
labels = []
session = requests.Session()
cache = {}

for shard_name, group in manifest.groupby("feature_shard"):
    print("Processing:", shard_name)

    label_shard_name = group.iloc[0]["label_shard"]

    if label_shard_name not in cache:
        response = session.get(BASE_URL + label_shard_name, timeout=60)
        response.raise_for_status()
        cache[label_shard_name] = np.load(
            io.BytesIO(response.content),
            allow_pickle=False,
        )

    shard_labels = cache[label_shard_name]

    for _, row in group.iterrows():
        index = int(row["row_in_shard"])
        expected_label_id = int(row["label_id"])

        if int(shard_labels[index]) != expected_label_id:
            raise ValueError(
                f"Label mismatch: {row['label']} row {index}"
            )

        start = DATA_OFFSET + index * FEATURE_BYTES
        end = start + FEATURE_BYTES - 1

        response = session.get(
            BASE_URL + shard_name,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=60,
        )
        response.raise_for_status()

        if response.status_code != 206:
            raise RuntimeError(
                f"Range request failed: HTTP {response.status_code}"
            )

        if len(response.content) != FEATURE_BYTES:
            raise RuntimeError(
                f"Unexpected bytes received: {len(response.content)}"
            )

        content_range = response.headers.get("Content-Range", "")
        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Unexpected range: {content_range}")

        sample = np.frombuffer(
            response.content,
            dtype=np.float16,
        ).reshape(200, 450).copy()

        samples.append(sample)
        labels.append(row["label"])

X = np.stack(samples)
y = np.asarray(labels)

np.savez_compressed(
    OUTPUT_DIR / "validation_asl.npz",
    X=X,
    y=y,
)

print("\nValidation extraction complete")
print("Features:", X.shape)
print("Labels:", y.shape)
print("Class counts:", dict(zip(*np.unique(y, return_counts=True))))
print("Saved:", OUTPUT_DIR / "validation_asl.npz")
