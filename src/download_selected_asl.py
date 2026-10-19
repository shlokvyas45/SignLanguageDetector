import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "asl_citizen"
OUTPUT_DIR = ROOT / "data" / "selected_asl"
BASE_URL = "https://huggingface.co/datasets/SharoonArshad/asl-citizen-processed-200/resolve/main/"

SELECTED_LABELS = {"HELLO", "MORNING", "NAME", "TWO", "NIGHT1"}
FEATURE_BYTES = 200 * 450 * 2
DATA_OFFSET = 128

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = pd.read_csv(DATA_DIR / "metadata" / "train_manifest.csv")
manifest = manifest[manifest["label"].isin(SELECTED_LABELS)]

samples = []
labels = []

session = requests.Session()

for shard_name, group in manifest.groupby("feature_shard"):
    print(f"Processing {shard_name}...")

    label_shard_name = group.iloc[0]["label_shard"]
    label_url = BASE_URL + label_shard_name

    response = session.get(label_url, timeout=60)
    response.raise_for_status()
    shard_labels = np.load(io.BytesIO(response.content), allow_pickle=False)

    feature_url = BASE_URL + shard_name

    for _, row in group.iterrows():
        index = int(row["row_in_shard"])
        expected_label_id = int(row["label_id"])

        if int(shard_labels[index]) != expected_label_id:
            raise ValueError(
                f"Label mismatch for {row['label']} at {shard_name}, row {index}"
            )

        start = DATA_OFFSET + index * FEATURE_BYTES
        end = start + FEATURE_BYTES - 1

        response = session.get(
            feature_url,
            headers={"Range": f"bytes={start}-{end}"},
            timeout=60,
        )
        response.raise_for_status()

        content_range = response.headers.get("Content-Range", "")
        if response.status_code != 206 or len(response.content) != FEATURE_BYTES:
            raise RuntimeError(
                f"Unexpected response for {shard_name}, row {index}: "
                f"status={response.status_code}, bytes={len(response.content)}"
            )

        if not content_range.startswith(f"bytes {start}-{end}/"):
            raise RuntimeError(f"Incorrect byte range returned: {content_range}")

        sample = np.frombuffer(response.content, dtype=np.float16)
        sample = sample.reshape(200, 450).copy()

        samples.append(sample)
        labels.append(row["label"])

X = np.stack(samples)
y = np.asarray(labels)

np.savez_compressed(OUTPUT_DIR / "selected_asl.npz", X=X, y=y)

print("\nExtraction complete!")
print("Samples:", len(y))
print("Feature shape:", X.shape)
print("Classes:", dict(zip(*np.unique(y, return_counts=True))))
print("Saved to:", OUTPUT_DIR / "selected_asl.npz")