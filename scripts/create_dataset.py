"""Create small VinDr datasets for UI testing from prepared data."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

# Config
SCRIPT_DIR = Path(__file__).parent.parent  # cm3070-final-project/
SOURCE_DIR = SCRIPT_DIR / "datasets/prep/vindr-whole"

PRESETS = {
    # 5 per class = 20 total
    "tiny": 5,
    # 100 per class = 400 total
    "small": 100,
    # 500 per class (~1000 train, ~400 test, capped by available malignant)
    "large": 500,
}


def main():
    parser = argparse.ArgumentParser(description="Create small VinDr dataset for UI testing")
    parser.add_argument(
            "--preset", choices=PRESETS.keys(), default="tiny",
            help="Dataset size: tiny (20 images), small (400 images), or large (~2000 images)")
    args = parser.parse_args()

    n_per_class = PRESETS[args.preset]
    output_dir = SCRIPT_DIR / f"datasets/prep/vindr-ui-{args.preset}"

    print(f"Output: {output_dir}")
    print()

    # Create output directories
    for split in ['train', 'test']:
        for label in ['benign', 'malignant']:
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    # Process train and test splits
    for split in ['train', 'test']:
        csv_path = SOURCE_DIR / f"{split}.csv"
        df = pd.read_csv(csv_path)

        for label, label_val in [('benign', 0), ('malignant', 1)]:
            available = len(df[df['label'] == label_val])
            n = min(n_per_class, available)

            label_df = df[df['label'] == label_val].sample(n=n, random_state=42)

            for idx, (_, row) in enumerate(label_df.iterrows()):
                src = SOURCE_DIR / "img" / row['filename']
                dst = output_dir / split / label / f"{idx:03d}.png"

                if src.exists():
                    shutil.copy(src, dst)

            print(f"{split}/{label}: {n} images")

    print()
    print(f"Done! Dataset created at: {output_dir}")


if __name__ == "__main__":
    main()
