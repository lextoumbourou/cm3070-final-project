"""
CBIS-DDSM dataset preparation script

This script prepares the CBIS-DDSM dataset for training by:
1. Loading mass and calcification case descriptions
2. Extracting ROI crops from the JPEG files
3. Resizing images to a fixed resolution (256x256)
4. Using the official test split and creating a validation split from training data
5. Saving processed images and metadata CSVs
"""

from pathlib import Path
from typing import Tuple
import argparse
import logging
from pydantic import BaseModel

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

RAW_DATA_ROOT = Path("datasets/cbis-ddsm-breast-cancer-image-dataset")
JPEG_ROOT = RAW_DATA_ROOT / "jpeg"
CSV_ROOT = RAW_DATA_ROOT / "csv"
OUTPUT_ROOT = Path("datasets/prep/cbis-ddsm")
IMG_OUTPUT_DIR = OUTPUT_ROOT / "img"
TARGET_SIZE = (256, 256)


def load_and_combine_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CBIS-DDSM CSVs and keep train/test splits separate

    Returns:
        Train DataFrame, Test DataFrame, and DICOM info DataFrame
    """
    logger.info("Loading CBIS-DDSM metadata files...")

    train_mass = pd.read_csv(CSV_ROOT / "mass_case_description_train_set.csv")
    test_mass = pd.read_csv(CSV_ROOT / "mass_case_description_test_set.csv")
    train_calc = pd.read_csv(CSV_ROOT / "calc_case_description_train_set.csv")
    test_calc = pd.read_csv(CSV_ROOT / "calc_case_description_test_set.csv")

    train_mass["abnormality_category"] = "mass"
    test_mass["abnormality_category"] = "mass"
    train_calc["abnormality_category"] = "calcification"
    test_calc["abnormality_category"] = "calcification"

    train_df = pd.concat([train_mass, train_calc], ignore_index=True)
    test_df = pd.concat([test_mass, test_calc], ignore_index=True)

    dicom_info_df = pd.read_csv(CSV_ROOT / "dicom_info.csv")

    logger.info(
        f"Train cases: {len(train_df)} (Mass: {len(train_mass)}, Calc: {len(train_calc)})"
    )
    logger.info(
        f"Test cases: {len(test_df)} (Mass: {len(test_mass)}, Calc: {len(test_calc)})"
    )
    logger.info(f"Train patients: {train_df['patient_id'].nunique()}")
    logger.info(f"Test patients: {test_df['patient_id'].nunique()}")

    return train_df, test_df, dicom_info_df


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


class ImageInfo(BaseModel):
    subject_id: str
    study_uid: str
    series_uid: str
    img_idx: int

def get_img_info_from_dcm_file(path: Path) -> ImageInfo:
    path_parts = str(path).split("/")
    subject_id = path_parts[0]
    study_uid = path_parts[1]
    series_uid = path_parts[2]
    img_idx = int(path_parts[-1].strip().replace(".dcm", "")[-1])

    return ImageInfo(
        subject_id=subject_id,
        study_uid=study_uid,
        series_uid=series_uid,
        img_idx=img_idx,
    )

def get_jpg_path(img_file_path: str):
    return JPEG_ROOT / img_file_path.replace("CBIS-DDSM/jpeg/", "")


def get_cropped_img_from_roi_path(mask_img_path: Path) -> Path:
    parent_dir = mask_img_path.parent 
    files = list(parent_dir.glob("*.jpg"))
    if not files:
        import pdb; pdb.set_trace()
    filtered_files = [f for f in files if f.name.startswith("2-")]
    if not filtered_files:
        import pdb; pdb.set_trace()
    crop = filtered_files[0]
    return crop

def process_case(
    row: pd.Series,
    dicom_info_df: pd.DataFrame,
    abnormality_category: str,
    output_dir: Path,
    case_idx: int,
) -> dict:
    """
    Process a single case: load image, extract ROI, resize, and save

    Args:
        row: Row from the case description CSV
        dicom_info_df: DICOM metadata DataFrame
        abnormality_category: 'mass' or 'calcification'
        output_dir: Directory to save processed images
        case_idx: Index to ensure unique filenames

    Returns:
        Dictionary with case metadata
    """
    # Get image path
    image_path_str = row["cropped image file path"]
    img_info = get_img_info_from_dcm_file(image_path_str)

    # Match with dicom_info to get the JPEG path
    dicom_row = dicom_info_df[
        (dicom_info_df["PatientID"] == img_info.subject_id)
        & (dicom_info_df["StudyInstanceUID"] == img_info.study_uid)
        & (dicom_info_df["SeriesInstanceUID"] == img_info.series_uid)
        & (dicom_info_df["SeriesDescription"] == 'cropped images')
    ]

    if len(dicom_row) != 1:
        logger.warning(f"No DICOM match found for: {image_path_str}")
        return None

    jpeg_path = get_jpg_path(dicom_row.iloc[0].image_path)
    if not jpeg_path.exists():
        logger.warning(f"Image file not found: {jpeg_path}")
        return None

    # Load and process image
    img = Image.open(jpeg_path).convert("L")  # Convert to grayscale

    # Resize to target size
    img_resized = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Create output filename
    patient_id = row["patient_id"]
    breast_side = row["left or right breast"]
    image_view = row["image view"]
    pathology = row["pathology"]

    # Create a unique filename with case index
    filename = f"{case_idx:05d}_{patient_id}_{breast_side}_{image_view}_{abnormality_category}_{pathology}.png"
    output_path = output_dir / filename

    # Save processed image
    img_resized.save(output_path)

    # Create metadata entry
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
    dicom_info_df: pd.DataFrame,
    img_output_dir: Path,
) -> pd.DataFrame:
    """
    Process all cases in a split and save metadata

    Args:
        split_df: DataFrame with cases to process
        split_name: Name of the split (train/val/test)
        dicom_info_df: DICOM metadata DataFrame
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
            row, dicom_info_df, abnormality_category, img_output_dir, idx
        )

        if metadata is not None:
            processed_cases.append(metadata)

    metadata_df = pd.DataFrame(processed_cases)

    # Save CSV to OUTPUT_ROOT
    csv_path = OUTPUT_ROOT / f"{split_name}.csv"
    metadata_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    logger.info(f"{split_name} - Total cases: {len(metadata_df)}")
    logger.info(f"{split_name} - Benign: {(metadata_df['label'] == 0).sum()}")
    logger.info(f"{split_name} - Malignant: {(metadata_df['label'] == 1).sum()}")

    return metadata_df


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

    train_all_df, test_df, dicom_info_df = load_and_combine_data()

    # Split training data into train and validation sets
    train_df, val_df = split_by_patient(
        train_all_df, val_ratio=args.val_ratio, random_state=args.random_seed
    )

    train_metadata = process_and_save_split(
        train_df, "train", dicom_info_df, IMG_OUTPUT_DIR
    )
    val_metadata = process_and_save_split(val_df, "val", dicom_info_df, IMG_OUTPUT_DIR)
    test_metadata = process_and_save_split(
        test_df, "test", dicom_info_df, IMG_OUTPUT_DIR
    )

    logger.info("CBIS-DDSM dataset preparation complete!")
    logger.info(f"Processed images saved to: {IMG_OUTPUT_DIR}")
    logger.info(f"Metadata CSVs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
