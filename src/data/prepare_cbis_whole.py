"""
CBIS-DDSM whole mammogram dataset preparation for end-to-end classification.

Prepares full mammograms for training whole-image classifiers following
Shen et al. (2019) approach:

- One image per unique mammogram (patient_id, breast_side, image_view)
- Binary labels: malignant if ANY abnormality in image is malignant
- Resised to consistent dimensions (default: 1152x896, same as paper)
- No cropping, preserves full breast context
"""

from pathlib import Path
from typing import Tuple, Optional, Dict, List
import argparse
import logging

import pandas as pd
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from src.data.cbis_ddsm import DCMData, parse_dcm_path, resolve_dcm_path, load_dicom_array
from src.data.preprocessing import get_breast_bbox, normalise_to_uint8


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default target size from Shen et al. (2019)
# Paper used 1152x896, downsized from original ~4000x3000
TARGET_WIDTH = 1152
TARGET_HEIGHT = 896

CSV_ROOT = Path.home() / "datasets/CBIS-DDSM/fixed-csv"
DATASET_ROOT = Path("datasets/CBIS-DDSM")
OUTPUT_ROOT = Path("datasets/prep/cbis-ddsm-whole")
IMG_OUTPUT_DIR = OUTPUT_ROOT / "img"


def load_and_combine_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CBIS-DDSM CSVs and keep train/test splits separate.
    """
    logger.info(f"Loading CBIS-DDSM metadata files from {CSV_ROOT}...")

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

    metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")

    logger.info(
        f"Train abnormalities: {len(train_df)} (Mass: {len(train_mass)}, Calc: {len(train_calc)})"
    )
    logger.info(
        f"Test abnormalities: {len(test_df)} (Mass: {len(test_mass)}, Calc: {len(test_calc)})"
    )
    logger.info(f"Train patients: {train_df['patient_id'].nunique()}")
    logger.info(f"Test patients: {test_df['patient_id'].nunique()}")

    return train_df, test_df, metadata_df


def split_by_patient(
    df: pd.DataFrame, val_ratio: float, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data at patient level to avoid data leakage.
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

    return train_df, val_df


def group_by_image(df: pd.DataFrame) -> Dict[tuple, pd.DataFrame]:
    """
    Group abnormalities by unique mammogram image.

    Each mammogram is identified by (patient_id, breast_side, image_view).
    Multiple abnormalities can exist in a single mammogram.
    """
    grouped = {}
    for _, row in df.iterrows():
        key = (row["patient_id"], row["left or right breast"], row["image view"])
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(row)

    # Convert lists to DataFrames
    return {key: pd.DataFrame(rows) for key, rows in grouped.items()}


def get_image_label(group_df: pd.DataFrame) -> int:
    """
    Determine binary label for a mammogram image.

    Label is 1 (malignant) if ANY abnormality in the image is malignant,
    otherwise 0 (benign/normal).
    """
    pathologies = group_df["pathology"].values
    for pathology in pathologies:
        if pathology == "MALIGNANT":
            return 1
    return 0


def get_abnormality_summary(group_df: pd.DataFrame) -> Dict:
    """
    Summarise abnormalities in an image for metadata.
    """
    categories = group_df["abnormality_category"].tolist()
    pathologies = group_df["pathology"].tolist()

    return {
        "num_abnormalities": len(group_df),
        "has_mass": "mass" in categories,
        "has_calc": "calcification" in categories,
        "has_malignant": "MALIGNANT" in pathologies,
        "has_benign": any(p in ("BENIGN", "BENIGN_WITHOUT_CALLBACK") for p in pathologies),
        "abnormality_types": ",".join(sorted(set(categories))),
        "pathology_types": ",".join(sorted(set(pathologies))),
    }


def preprocess_mammogram(
    img: np.ndarray,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    crop_breast: bool = False
) -> np.ndarray:
    """Preprocess full mammogram: optionally crop to breast region, normalise and resize."""
    if crop_breast:
        x, y, w, h = get_breast_bbox(img)
        img = img[y:y+h, x:x+w]

    img_normalized = normalise_to_uint8(img)
    return cv2.resize(img_normalized, (target_width, target_height))


def load_full_image(row: pd.Series, metadata_df: pd.DataFrame) -> Optional[np.ndarray]:
    """Load full mammogram image from DICOM."""
    image_path_str = row["image file path"]
    try:
        dcm_data = parse_dcm_path(image_path_str)
        dicom_path = resolve_dcm_path(dcm_data, metadata_df, DATASET_ROOT)
        if not dicom_path.exists():
            logger.warning(f"DICOM file not found: {dicom_path}")
            return None
        img = load_dicom_array(dicom_path)
        return img
    except IndexError:
        logger.warning(f"No metadata match for: {image_path_str}")
    except Exception as e:
        logger.warning(f"Failed to load image {image_path_str}: {type(e).__name__}: {e}")
    return None


def process_image(
    image_key: tuple,
    group_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    image_idx: int,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    crop_breast: bool = False,
) -> Optional[dict]:
    """
    Process a single mammogram image.
    """
    patient_id, breast_side, image_view = image_key

    first_row = group_df.iloc[0]
    img = load_full_image(first_row, metadata_df)
    if img is None:
        return None

    img_processed = preprocess_mammogram(img, target_width, target_height, crop_breast)

    # Determine label (malignant if ANY abnormality is malignant)
    label = get_image_label(group_df)

    # Generate filename
    label_str = "MAL" if label == 1 else "BEN"
    filename = f"{image_idx:05d}_{patient_id}_{breast_side}_{image_view}_{label_str}.png"
    output_path = output_dir / filename

    # Save image
    cv2.imwrite(str(output_path), img_processed)

    # Get abnormality summary
    summary = get_abnormality_summary(group_df)

    return {
        "filename": filename,
        "patient_id": patient_id,
        "breast_side": breast_side,
        "image_view": image_view,
        "label": label,
        "num_abnormalities": summary["num_abnormalities"],
        "has_mass": summary["has_mass"],
        "has_calc": summary["has_calc"],
        "has_malignant": summary["has_malignant"],
        "has_benign": summary["has_benign"],
        "abnormality_types": summary["abnormality_types"],
        "pathology_types": summary["pathology_types"],
    }


def process_and_save_split(
    split_df: pd.DataFrame,
    split_name: str,
    metadata_df: pd.DataFrame,
    img_output_dir: Path,
    output_root: Path,
    target_width: int = TARGET_WIDTH,
    target_height: int = TARGET_HEIGHT,
    crop_breast: bool = False,
    start_idx: int = 0
) -> Tuple[pd.DataFrame, int]:
    """
    Process all images in a split and save metadata.
    """
    logger.info(f"Processing {split_name} split...")

    grouped = group_by_image(split_df)
    logger.info(f"{split_name}: {len(grouped)} unique images from {len(split_df)} abnormalities")

    processed_images = []
    current_idx = start_idx

    for image_key, group_df in tqdm(
        grouped.items(), total=len(grouped), desc=f"Processing {split_name}"
    ):
        metadata = process_image(
            image_key, group_df, metadata_df, img_output_dir, current_idx,
            target_width=target_width, target_height=target_height,
            crop_breast=crop_breast
        )

        if metadata is not None:
            processed_images.append(metadata)
            current_idx += 1

    result_df = pd.DataFrame(processed_images)

    # Save CSV
    csv_path = output_root / f"{split_name}.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    # Log statistics
    if len(result_df) > 0:
        n_malignant = (result_df["label"] == 1).sum()
        n_benign = (result_df["label"] == 0).sum()
        logger.info(f"{split_name} - Total images: {len(result_df)}")
        logger.info(f"{split_name} - Malignant: {n_malignant} ({100*n_malignant/len(result_df):.1f}%)")
        logger.info(f"{split_name} - Benign: {n_benign} ({100*n_benign/len(result_df):.1f}%)")

    return result_df, current_idx


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CBIS-DDSM whole mammogram dataset for end-to-end classification"
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
        "--target-width",
        type=int,
        default=TARGET_WIDTH,
        help=f"Target image width in pixels (default: {TARGET_WIDTH})",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=TARGET_HEIGHT,
        help=f"Target image height in pixels (default: {TARGET_HEIGHT})",
    )
    parser.add_argument(
        "--crop",
        action="store_true",
        help="Crop to breast region using Otsu thresholding before resizing",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: datasets/prep/cbis-ddsm-whole or cbis-ddsm-whole-crop)",
    )

    args = parser.parse_args()

    if args.output_dir:
        output_root = Path(args.output_dir)
    else:
        output_root = Path("datasets/prep/cbis-ddsm-whole-crop") if args.crop else OUTPUT_ROOT
    img_output_dir = output_root / "img"

    logger.info("Starting CBIS-DDSM whole mammogram dataset preparation...")
    logger.info(f"Target size: {args.target_width}x{args.target_height}")
    logger.info(f"Crop breast region: {args.crop}")
    logger.info(f"Output directory: {output_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    img_output_dir.mkdir(parents=True, exist_ok=True)

    train_all_df, test_df, metadata_df = load_and_combine_data()

    # Split training data into train and validation sets (patient-level)
    train_df, val_df = split_by_patient(
        train_all_df, val_ratio=args.val_ratio, random_state=args.random_seed
    )

    train_metadata, next_idx = process_and_save_split(
        train_df, "train", metadata_df, img_output_dir, output_root,
        target_width=args.target_width, target_height=args.target_height,
        crop_breast=args.crop, start_idx=0
    )

    val_metadata, next_idx = process_and_save_split(
        val_df, "val", metadata_df, img_output_dir, output_root,
        target_width=args.target_width, target_height=args.target_height,
        crop_breast=args.crop, start_idx=next_idx
    )

    test_metadata, _ = process_and_save_split(
        test_df, "test", metadata_df, img_output_dir, output_root,
        target_width=args.target_width, target_height=args.target_height,
        crop_breast=args.crop, start_idx=next_idx
    )

    logger.info("Dataset preparation complete!")
    total = len(train_metadata) + len(val_metadata) + len(test_metadata)
    logger.info(f"Total images: {total}")
    logger.info(f"  Train: {len(train_metadata)}")
    logger.info(f"  Val: {len(val_metadata)}")
    logger.info(f"  Test: {len(test_metadata)}")


if __name__ == "__main__":
    main()
