"""Dataset classes for training and inference."""

import csv
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlxim.data._base import Dataset
from PIL import Image


class CSVDataset(Dataset):
    """Loads images from a CSV file with filename and label columns."""

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform=None,
    ):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        with open(csv_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["filename"], int(row["label"])))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filename, label = self.samples[idx]
        img_path = self.img_dir / filename

        img = Image.open(img_path).convert("RGB")
        img = np.array(img).astype(np.uint8)

        if self.transform:
            img = self.transform(image=img)["image"]

        img = img.astype(np.float32) / 255.0
        return mx.array(img), mx.array(label)
