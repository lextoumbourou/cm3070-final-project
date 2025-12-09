"""
CBIS-DDSM dataset preparation script

This script prepares the CBIS-DDSM dataset for training by:
1. Loading mass and calcification case descriptions
2. Preprocessing full mammogram images (crop and resize to grayscale)
3. Using the official test split and creating a validation split from training data
4. Saving processed images and metadata CSVs
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
from sklearn.model_selection import train_test_split
from pydantic import BaseModel


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_SIZE = 512

RAW_DATA_ROOT = Path("datasets/CBIS-DDSM")
IMG_ROOT = RAW_DATA_ROOT / "CBIS-DDSM"
OUTPUT_ROOT = Path("datasets/prep/cbis-ddsm")
IMG_OUTPUT_DIR = OUTPUT_ROOT / "img"


class DCMData(BaseModel):
    """Parsed DICOM file path data."""
    subject_id: str
    study_uid: str
    series_uid: str


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


def get_file_data_from_dcm(dcm_path: str) -> DCMData:
    """Parse DICOM file path to extract subject_id, study_uid, and series_uid."""
    data = str(dcm_path).strip().split("/")
    return DCMData(subject_id=data[0], study_uid=data[1], series_uid=data[2])


def get_meta_from_dcm_data(dcm_data: DCMData, metadata_df: pd.DataFrame) -> pd.Series:
    """Look up metadata row from parsed DCM data."""
    meta = metadata_df[
        (metadata_df["Subject ID"] == dcm_data.subject_id) &
        (metadata_df["Series UID"] == dcm_data.series_uid) &
        (metadata_df["Study UID"] == dcm_data.study_uid)
    ].iloc[0]
    return meta


def get_img_from_file_location(file_location: Path) -> Path:
    """Get the DICOM file from a directory."""
    files = list(file_location.glob("*.dcm"))
    return files[0]


def dicom_to_array(file_path: Path) -> np.ndarray:
    """Load a DICOM file and return as numpy array."""
    ds = pydicom.dcmread(file_path)
    return ds.pixel_array


def get_img_path(img_path_str: str, metadata_df: pd.DataFrame) -> Path:
    """Get the full image path from the image file path string."""
    dcm_data = get_file_data_from_dcm(img_path_str)
    img_meta = get_meta_from_dcm_data(dcm_data, metadata_df)
    return get_img_from_file_location(RAW_DATA_ROOT / img_meta["File Location"])


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

def process_case(
    row: pd.Series,
    metadata_df: pd.DataFrame,
    abnormality_category: str,
    output_dir: Path,
    case_idx: int,
) -> dict:
    image_path_str = row["image file path"]

    try:
        dicom_path = get_img_path(image_path_str, metadata_df)
    except (IndexError, KeyError):
        logger.warning(f"No metadata match found for: {image_path_str}")
        return None

    if not dicom_path.exists():
        logger.warning(f"Image file not found: {dicom_path}")
        return None

    try:
        img = dicom_to_array(dicom_path)
        img_processed = preprocess_mammogram(img)
    except Exception as e:
        logger.warning(f"Failed to preprocess {dicom_path}: {e}")
        return None

    patient_id = row["patient_id"]
    breast_side = row["left or right breast"]
    image_view = row["image view"]
    pathology = row["pathology"]

    filename = f"{case_idx:05d}_{patient_id}_{breast_side}_{image_view}_{abnormality_category}_{pathology}.png"
    output_path = output_dir / filename

    cv2.imwrite(str(output_path), img_processed)

    label = 1 if pathology == "MALIGNANT" else 0

    metadata = {
        "filename": filename,
        "patient_id": patient_id,
        "breast_side": breast_side,
        "image_view": image_view,
        "abnormality_category": abnormality_category,
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
) -> pd.DataFrame:
    """
    Process all cases in a split and save metadata

    Args:
        split_df: DataFrame with cases to process
        split_name: Name of the split (train/val/test)
        metadata_df: Metadata DataFrame for image lookup
        img_output_dir: Directory to save processed images

    Returns:
        DataFrame with processed case metadata
    """
    logger.info(f"Processing {split_name} split...")

    processed_cases = []

    for idx, row in tqdm(
        split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"
    ):
        abnormality_category = row["abnormality_category"]
        metadata = process_case(
            row, metadata_df, abnormality_category, img_output_dir, idx
        )

        if metadata is not None:
            processed_cases.append(metadata)

    result_df = pd.DataFrame(processed_cases)

    # Save CSV to OUTPUT_ROOT
    csv_path = OUTPUT_ROOT / f"{split_name}.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    logger.info(f"{split_name} - Total cases: {len(result_df)}")
    logger.info(f"{split_name} - Benign: {(result_df['label'] == 0).sum()}")
    logger.info(f"{split_name} - Malignant: {(result_df['label'] == 1).sum()}")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset")
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

    args = parser.parse_args()

    logger.info("Starting CBIS-DDSM dataset preparation...")
    logger.info(f"Target image size: {TARGET_SIZE}")
    logger.info(
        f"Using official test split and creating validation split with ratio: {args.val_ratio}"
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_all_df, test_df, metadata_df = load_and_combine_data()

    # Split training data into train and validation sets
    train_df, val_df = split_by_patient(
        train_all_df, val_ratio=args.val_ratio, random_state=args.random_seed
    )

    train_metadata = process_and_save_split(
        train_df, "train", metadata_df, IMG_OUTPUT_DIR
    )
    val_metadata = process_and_save_split(val_df, "val", metadata_df, IMG_OUTPUT_DIR)
    test_metadata = process_and_save_split(
        test_df, "test", metadata_df, IMG_OUTPUT_DIR
    )

    logger.info("CBIS-DDSM dataset preparation complete!")
    logger.info(f"Processed images saved to: {IMG_OUTPUT_DIR}")
    logger.info(f"Metadata CSVs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
