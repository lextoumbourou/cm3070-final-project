"""
CBIS-DDSM dataset preparation script

This script prepares the CBIS-DDSM dataset for training by:
1. Loading mass and calcification case descriptions
2. Preprocessing images (crop and resize to grayscale)
3. Using the official test split and creating a validation split from training data
4. Saving processed images and metadata CSVs

Supports two modes:
- full: Extract full mammograms with automatic breast region detection
- roi: Extract pre-cropped ROI abnormality images (higher resolution on lesion)
"""

from pathlib import Path
from typing import Tuple
import argparse
import logging

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.cbis_ddsm import DCMData, parse_dcm_path, resolve_dcm_path, load_dicom_array


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_SIZE = 512

RAW_DATA_ROOT = Path("datasets/CBIS-DDSM")
IMG_ROOT = RAW_DATA_ROOT / "CBIS-DDSM"
OUTPUT_ROOT = Path("datasets/prep/cbis-ddsm")
IMG_OUTPUT_DIR = OUTPUT_ROOT / "img"


def load_and_combine_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CBIS-DDSM CSVs and keep train/test splits separate

    Returns:
        Train DataFrame, Test DataFrame, and metadata DataFrame
    """
    logger.info("Loading CBIS-DDSM metadata files...")

    train_mass = pd.read_csv(RAW_DATA_ROOT / "mass_case_description_train_set.csv")
    test_mass = pd.read_csv(RAW_DATA_ROOT / "mass_case_description_test_set.csv")
    train_calc = pd.read_csv(RAW_DATA_ROOT / "calc_case_description_train_set.csv")
    test_calc = pd.read_csv(RAW_DATA_ROOT / "calc_case_description_test_set.csv")

    train_mass["abnormality_category"] = "mass"
    test_mass["abnormality_category"] = "mass"
    train_calc["abnormality_category"] = "calcification"
    test_calc["abnormality_category"] = "calcification"

    train_df = pd.concat([train_mass, train_calc], ignore_index=True)
    test_df = pd.concat([test_mass, test_calc], ignore_index=True)

    metadata_df = pd.read_csv(RAW_DATA_ROOT / "metadata.csv")

    logger.info(
        f"Train cases: {len(train_df)} (Mass: {len(train_mass)}, Calc: {len(train_calc)})"
    )
    logger.info(
        f"Test cases: {len(test_df)} (Mass: {len(test_mass)}, Calc: {len(test_calc)})"
    )
    logger.info(f"Train patients: {train_df['patient_id'].nunique()}")
    logger.info(f"Test patients: {test_df['patient_id'].nunique()}")

    return train_df, test_df, metadata_df


def split_by_patient(
    df: pd.DataFrame, val_ratio: float, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data at patient level to avoid data leakage

    Args:
        df: DataFrame to split
        val_ratio: Proportion for validation set
        random_state: Random seed for reproducibility

    Returns:
        Train and validation DataFrames
    """
    unique_patients = df["patient_id"].unique()

    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_ratio, random_state=random_state
    )

    train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(val_patients)].reset_index(drop=True)

    logger.info(
        f"Split patients - Train: {len(train_patients)}, Val: {len(val_patients)}"
    )
    logger.info(f"Split cases - Train: {len(train_df)}, Val: {len(val_df)}")

    return train_df, val_df


def get_filepath_from_dcm_data(dcm_data: DCMData, metadata_df: pd.DataFrame) -> Path:
    """Resolve DICOM file path using metadata lookup."""
    return resolve_dcm_path(dcm_data, metadata_df, RAW_DATA_ROOT)


def apply_morphological_transforms(thresh_frame, iterations: int = 2):
    kernel = np.ones((100, 100), np.uint8)
    opened_mask = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed_mask


def get_contours_from_mask(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


def crop_coords(img: np.ndarray):
    """Get bounding box coordinates for breast region using thresholding."""
    # Normalize to uint8 for OpenCV operations
    if img.dtype != np.uint8:
        # Normalize to 0-255 range
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img

    blur = cv2.GaussianBlur(img_normalized, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph_img = apply_morphological_transforms(breast_mask)
    return get_contours_from_mask(morph_img)


def preprocess_mammogram(img: np.ndarray, target_size=TARGET_SIZE):
    """Preprocess mammogram image: crop and resize to grayscale."""
    x, y, w, h = crop_coords(img)
    img_cropped = img[y:y+h, x:x+w]

    # Normalize to 0-255 range for saving as PNG
    if img_cropped.dtype != np.uint8:
        img_normalized = ((img_cropped - img_cropped.min()) / (img_cropped.max() - img_cropped.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img_cropped

    img_final = cv2.resize(img_normalized, (target_size, target_size))
    return img_final


def preprocess_roi(img: np.ndarray, target_size=TARGET_SIZE):
    """Preprocess ROI image: normalize and resize (no cropping needed)."""
    # Normalize to 0-255 range for saving as PNG
    if img.dtype != np.uint8:
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img

    img_final = cv2.resize(img_normalized, (target_size, target_size))
    return img_final


def process_case(
    row: pd.Series,
    metadata_df: pd.DataFrame,
    abnormality_category: str,
    output_dir: Path,
    case_idx: int,
    mode: str = "full",
    target_size: int = TARGET_SIZE,
) -> dict:
    """
    Process a single case.

    Args:
        mode: "full" for full mammogram with breast extraction,
              "roi" for pre-cropped ROI abnormality images
    """
    if mode == "full":
        image_path_str = row["image file path"]
    else:  # roi mode
        image_path_str = row["cropped image file path"]

    try:
        dcm_data = parse_dcm_path(image_path_str)
        dicom_path = get_filepath_from_dcm_data(dcm_data, metadata_df)
    except (IndexError, KeyError) as e:
        logger.warning(f"No metadata match found for: {image_path_str} - {e}")
        return None

    if not dicom_path.exists():
        logger.warning(f"DICOM file not found: {dicom_path}")
        return None

    try:
        img = load_dicom_array(dicom_path)
        if mode == "full":
            img_processed = preprocess_mammogram(img, target_size)
        else:  # roi mode
            img_processed = preprocess_roi(img, target_size)
    except Exception as e:
        logger.warning(f"Failed to preprocess {dicom_path}: {e}")
        return None

    patient_id = row["patient_id"]
    breast_side = row["left or right breast"]
    image_view = row["image view"]
    pathology = row["pathology"]
    abnormality_id = row.get("abnormality id", 1)

    filename = f"{case_idx:05d}_{patient_id}_{breast_side}_{image_view}_{abnormality_category}_{abnormality_id}_{pathology}.png"
    output_path = output_dir / filename

    cv2.imwrite(str(output_path), img_processed)

    label = 1 if pathology == "MALIGNANT" else 0

    metadata = {
        "filename": filename,
        "patient_id": patient_id,
        "breast_side": breast_side,
        "image_view": image_view,
        "abnormality_category": abnormality_category,
        "abnormality_id": abnormality_id,
        "pathology": pathology,
        "label": label,
        "assessment": row.get("assessment", None),
        "subtlety": row.get("subtlety", None),
    }

    # Add mass-specific or calc-specific features
    if abnormality_category == "mass":
        metadata["mass_shape"] = row.get("mass shape", None)
        metadata["mass_margins"] = row.get("mass margins", None)
    else:
        metadata["calc_type"] = row.get("calc type", None)
        metadata["calc_distribution"] = row.get("calc distribution", None)

    return metadata


def process_and_save_split(
    split_df: pd.DataFrame,
    split_name: str,
    metadata_df: pd.DataFrame,
    img_output_dir: Path,
    output_root: Path,
    mode: str = "full",
    target_size: int = TARGET_SIZE,
) -> pd.DataFrame:
    """
    Process all cases in a split and save metadata

    Args:
        split_df: DataFrame with cases to process
        split_name: Name of the split (train/val/test)
        metadata_df: Metadata DataFrame for image lookup
        img_output_dir: Directory to save processed images
        output_root: Root directory for CSV output
        mode: "full" or "roi"
        target_size: Target image size

    Returns:
        DataFrame with processed case metadata
    """
    logger.info(f"Processing {split_name} split ({mode} mode)...")

    processed_cases = []

    for idx, row in tqdm(
        split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"
    ):
        abnormality_category = row["abnormality_category"]
        metadata = process_case(
            row, metadata_df, abnormality_category, img_output_dir, idx,
            mode=mode, target_size=target_size
        )

        if metadata is not None:
            processed_cases.append(metadata)

    result_df = pd.DataFrame(processed_cases)

    # Save CSV to output_root
    csv_path = output_root / f"{split_name}.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    logger.info(f"{split_name} - Total cases: {len(result_df)}")
    logger.info(f"{split_name} - Benign: {(result_df['label'] == 0).sum()}")
    logger.info(f"{split_name} - Malignant: {(result_df['label'] == 1).sum()}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "roi"],
        default="full",
        help="Extraction mode: 'full' for full mammograms with breast detection, "
             "'roi' for pre-cropped ROI abnormality images (default: full)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of training patients for validation (default: 0.1)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=TARGET_SIZE,
        help=f"Target image size in pixels (default: {TARGET_SIZE})",
    )

    args = parser.parse_args()

    # Set output directory based on mode
    if args.mode == "full":
        output_root = Path("datasets/prep/cbis-ddsm")
    else:
        output_root = Path("datasets/prep/cbis-ddsm-roi")

    img_output_dir = output_root / "img"

    logger.info(f"Starting CBIS-DDSM dataset preparation ({args.mode} mode)...")
    logger.info(f"Target image size: {args.target_size}")
    logger.info(f"Output directory: {output_root}")
    logger.info(
        f"Using official test split and creating validation split with ratio: {args.val_ratio}"
    )

    output_root.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    train_all_df, test_df, metadata_df = load_and_combine_data()

    # Split training data into train and validation sets
    train_df, val_df = split_by_patient(
        train_all_df, val_ratio=args.val_ratio, random_state=args.random_seed
    )

    train_metadata = process_and_save_split(
        train_df, "train", metadata_df, img_output_dir, output_root,
        mode=args.mode, target_size=args.target_size
    )
    val_metadata = process_and_save_split(
        val_df, "val", metadata_df, img_output_dir, output_root,
        mode=args.mode, target_size=args.target_size
    )
    test_metadata = process_and_save_split(
        test_df, "test", metadata_df, img_output_dir, output_root,
        mode=args.mode, target_size=args.target_size
    )

    logger.info("CBIS-DDSM dataset preparation complete!")
    logger.info(f"Processed images saved to: {img_output_dir}")
    logger.info(f"Metadata CSVs saved to: {output_root}")


if __name__ == "__main__":
    main()
