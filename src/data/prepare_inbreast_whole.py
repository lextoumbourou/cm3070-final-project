"""
INbreast whole mammogram dataset preparation for end-to-end classification.

Prepares full mammograms matching the CBIS-DDSM whole format (1152x896)
for domain shift experiments with the Shen et al. pipeline.

Todo: refactor and reduce duplication between prepare_cbis_whole.py
"""

from pathlib import Path
from typing import Tuple
import argparse
import logging

import pandas as pd
import numpy as np
import cv2
import pydicom
from tqdm import tqdm

from src.data.preprocessing import normalise_to_uint8, get_breast_bbox


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_WIDTH = 1152
TARGET_HEIGHT = 896

RAW_DATA_ROOT = Path("datasets/INbreast Release 1.0")
DICOM_DIR = RAW_DATA_ROOT / "AllDICOMs"
OUTPUT_ROOT = Path("datasets/prep/inbreast-whole")
IMG_OUTPUT_DIR = OUTPUT_ROOT / "img"


def load_metadata() -> pd.DataFrame:
    logger.info("Loading INbreast metadata...")
    df = pd.read_csv(RAW_DATA_ROOT / "INbreast.csv", sep=";")
    df["pathology"] = df["Bi-Rads"].apply(classify_birads)
    df = df[df["pathology"] != "Unknown"].reset_index(drop=True)

    logger.info(f"Total images: {len(df)}")
    logger.info(f"Unique patients: {df['Patient ID'].nunique()}")
    logger.info(f"Benign: {(df['pathology'] == 'BENIGN').sum()}")
    logger.info(f"Malignant: {(df['pathology'] == 'MALIGNANT').sum()}")

    return df


def classify_birads(birads) -> str:
    birads_str = str(birads).strip()
    if birads_str in ["1", "2", "3"]:
        return "BENIGN"
    elif birads_str in ["4a", "4b", "4c", "4", "5", "6"]:
        return "MALIGNANT"
    return "Unknown"


def split_data(
    df: pd.DataFrame, test_ratio: float, val_ratio: float, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_df = df_shuffled[:n_test].reset_index(drop=True)
    val_df = df_shuffled[n_test:n_test + n_val].reset_index(drop=True)
    train_df = df_shuffled[n_test + n_val:].reset_index(drop=True)

    logger.info(f"Split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


def preprocess_mammogram(
    img_array: np.ndarray,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    crop_breast: bool = False
) -> np.ndarray:
    if crop_breast:
        x, y, w, h = get_breast_bbox(img_array)
        img_array = img_array[y:y+h, x:x+w]

    img_normalized = normalise_to_uint8(img_array)
    img_resized = cv2.resize(img_normalized, (target_width, target_height))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
    return img_rgb


def process_case(
    row: pd.Series,
    output_dir: Path,
    case_idx: int,
    target_width: int,
    target_height: int,
    crop_breast: bool
) -> dict:
    file_name = str(row["File Name"])
    dicom_files = list(DICOM_DIR.rglob(f"{file_name}*.dcm"))

    if not dicom_files:
        logger.warning(f"DICOM not found for: {file_name}")
        return None

    try:
        dicom_data = pydicom.dcmread(dicom_files[0])
        img_array = dicom_data.pixel_array
        img_processed = preprocess_mammogram(img_array, target_width, target_height, crop_breast)
    except Exception as e:
        logger.warning(f"Failed to preprocess {file_name}: {e}")
        return None

    patient_id = row["Patient ID"]
    laterality = row["Laterality"]
    view = row["View"]
    pathology = row["pathology"]
    birads = row["Bi-Rads"]

    label_str = "MAL" if pathology == "MALIGNANT" else "BEN"
    filename = f"{case_idx:05d}_{patient_id}_{laterality}_{view}_{label_str}.png"
    output_path = output_dir / filename

    cv2.imwrite(str(output_path), cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))

    label = 1 if pathology == "MALIGNANT" else 0

    return {
        "filename": filename,
        "patient_id": patient_id,
        "laterality": laterality,
        "view": view,
        "birads": birads,
        "pathology": pathology,
        "label": label,
    }


def process_and_save_split(
    split_df: pd.DataFrame,
    split_name: str,
    img_output_dir: Path,
    output_root: Path,
    target_width: int,
    target_height: int,
    crop_breast: bool,
    start_idx: int = 0
) -> Tuple[pd.DataFrame, int]:
    logger.info(f"Processing {split_name} split...")

    processed_cases = []
    current_idx = start_idx

    for _, row in tqdm(split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"):
        metadata = process_case(row, img_output_dir, current_idx, target_width, target_height, crop_breast)
        if metadata is not None:
            processed_cases.append(metadata)
            current_idx += 1

    metadata_df = pd.DataFrame(processed_cases)

    csv_path = output_root / f"{split_name}.csv"
    metadata_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    if len(metadata_df) > 0:
        n_malignant = (metadata_df['label'] == 1).sum()
        n_benign = (metadata_df['label'] == 0).sum()
        logger.info(f"{split_name} - Total: {len(metadata_df)}, Benign: {n_benign}, Malignant: {n_malignant}")

    return metadata_df, current_idx


def main():
    parser = argparse.ArgumentParser(description="Prepare INbreast whole mammogram dataset")
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--target-width", type=int, default=TARGET_WIDTH)
    parser.add_argument("--target-height", type=int, default=TARGET_HEIGHT)
    parser.add_argument("--crop", action="store_true", help="Crop to breast region before resizing")
    parser.add_argument("--output-dir", type=str, default=None)

    args = parser.parse_args()

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = OUTPUT_ROOT
    img_output_dir = output_root / "img"

    logger.info("Starting INbreast whole mammogram dataset preparation...")
    logger.info(f"Target size: {args.target_width}x{args.target_height}")
    logger.info(f"Crop breast: {args.crop}")
    logger.info(f"Output directory: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    df = load_metadata()

    train_df, val_df, test_df = split_data(
        df,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_seed,
    )

    train_metadata, next_idx = process_and_save_split(
        train_df, "train", img_output_dir, output_root,
        args.target_width, args.target_height, args.crop, start_idx=0
    )

    val_metadata, next_idx = process_and_save_split(
        val_df, "val", img_output_dir, output_root,
        args.target_width, args.target_height, args.crop, start_idx=next_idx
    )

    test_metadata, _ = process_and_save_split(
        test_df, "test", img_output_dir, output_root,
        args.target_width, args.target_height, args.crop, start_idx=next_idx
    )

    logger.info("Dataset preparation complete!")
    total = len(train_metadata) + len(val_metadata) + len(test_metadata)
    logger.info(f"Total images: {total}")
    logger.info(f"  Train: {len(train_metadata)}")
    logger.info(f"  Val: {len(val_metadata)}")
    logger.info(f"  Test: {len(test_metadata)}")


if __name__ == "__main__":
    main()
