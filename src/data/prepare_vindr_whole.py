"""
VinDr-Mammo whole mammogram preparation for inference and fine-tuning.

Outputs format compatible with inference_whole_image.py and finetune_whole_image.py:
- datasets/prep/vindr-whole/img/ (all images)
- datasets/prep/vindr-whole/train.csv, val.csv, test.csv
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

RAW_DATA_ROOT = Path("datasets/vindr-mammogram-dataset-dicom-to-png")
IMAGES_DIR = RAW_DATA_ROOT / "images_png"
CSV_PATH = RAW_DATA_ROOT / "vindr_detection_v1_folds.csv"

OUTPUT_ROOT = Path("datasets/prep/vindr-whole")

TARGET_WIDTH = 1152
TARGET_HEIGHT = 896


def classify_birads(birads_str):
    if birads_str in ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3']:
        return 'BENIGN'
    elif birads_str in ['BI-RADS 4', 'BI-RADS 5']:
        return 'MALIGNANT'
    return 'UNKNOWN'


def load_metadata():
    logger.info(f"Loading metadata from {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

    unique_images = df.drop_duplicates(subset=['image_id']).copy()
    unique_images['pathology'] = unique_images['breast_birads'].apply(classify_birads)
    unique_images = unique_images[unique_images['pathology'] != 'UNKNOWN']

    logger.info(f"Total unique images: {len(unique_images)}")
    logger.info(f"  Benign: {(unique_images['pathology'] == 'BENIGN').sum()}")
    logger.info(f"  Malignant: {(unique_images['pathology'] == 'MALIGNANT').sum()}")

    for split in unique_images['split'].unique():
        split_df = unique_images[unique_images['split'] == split]
        n_ben = (split_df['pathology'] == 'BENIGN').sum()
        n_mal = (split_df['pathology'] == 'MALIGNANT').sum()
        logger.info(f"  {split}: {len(split_df)} images (benign={n_ben}, malignant={n_mal})")

    return unique_images


def process_image(row, img_output_dir, target_width, target_height, idx):
    patient_id = row['patient_id']
    image_id = row['image_id']
    pathology = row['pathology']

    img_path = IMAGES_DIR / patient_id / image_id
    if not img_path.exists():
        return None

    try:
        img = Image.open(img_path)
        img_array = np.array(img)

        if len(img_array.shape) == 2 or len(img_array.shape) == 3 and img_array.shape[2] == 1:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)

        img_resized = cv2.resize(img_array, (target_width, target_height))

        label_str = "MAL" if pathology == "MALIGNANT" else "BEN"
        filename = f"{idx:05d}_{patient_id}_{label_str}.png"

        out_path = img_output_dir / filename
        cv2.imwrite(str(out_path), cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR))

        return {
            'filename': filename,
            'patient_id': patient_id,
            'image_id': image_id,
            'laterality': row['laterality'],
            'view': row['view'],
            'birads': row['breast_birads'],
            'pathology': pathology,
            'label': 1 if pathology == 'MALIGNANT' else 0,
        }
    except Exception as e:
        logger.warning(f"Failed to process {img_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Prepare VinDr whole mammogram dataset")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_ROOT))
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.15,
                        help="Split training into train/val")
    parser.add_argument("--balance", action="store_true",
                        help="Balance classes by downsampling majority class")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    img_output_dir = output_dir / "img"
    output_dir.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting VinDr whole mammogram preparation")
    logger.info(f"Target size: {args.target_width}x{args.target_height}")

    df = load_metadata()

    if args.max_images:
        df = df.head(args.max_images)
        logger.info(f"Limited to {args.max_images} images")

    train_df = df[df['split'] == 'training'].copy()
    test_df = df[df['split'] == 'test'].copy()

    def balance_df(split_df, name):
        benign = split_df[split_df['pathology'] == 'BENIGN']
        malignant = split_df[split_df['pathology'] == 'MALIGNANT']
        n_minority = min(len(benign), len(malignant))
        benign_sampled = benign.sample(n=n_minority, random_state=42)
        malignant_sampled = malignant.sample(n=n_minority, random_state=42)
        balanced = pd.concat([benign_sampled, malignant_sampled])
        logger.info(f"  {name}: {len(split_df)} → {len(balanced)} (balanced to {n_minority} per class)")
        return balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    if args.balance:
        logger.info("Balancing classes...")
        train_df = balance_df(train_df, "train")
        test_df = balance_df(test_df, "test")

    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val = int(len(train_df) * args.val_ratio)
    val_df = train_df[:n_val].copy()
    train_df = train_df[n_val:].copy()

    logger.info(f"Final split sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    idx = 0
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        logger.info(f"Processing {split_name} split...")
        processed = []

        for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=split_name):
            result = process_image(row, img_output_dir, args.target_width, args.target_height, idx)
            if result:
                processed.append(result)
                idx += 1

        if processed:
            metadata_df = pd.DataFrame(processed)
            csv_path = output_dir / f"{split_name}.csv"
            metadata_df.to_csv(csv_path, index=False)

            n_mal = (metadata_df['label'] == 1).sum()
            n_ben = (metadata_df['label'] == 0).sum()
            logger.info(f"  Saved {len(processed)} images (benign={n_ben}, malignant={n_mal})")

    logger.info("\nDataset preparation complete!")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
