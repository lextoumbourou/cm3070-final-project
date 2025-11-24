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
from typing import Tuple, List
import argparse
import logging

import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
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

    train_mass['abnormality_category'] = 'mass'
    test_mass['abnormality_category'] = 'mass'
    train_calc['abnormality_category'] = 'calcification'
    test_calc['abnormality_category'] = 'calcification'

    train_df = pd.concat([train_mass, train_calc], ignore_index=True)
    test_df = pd.concat([test_mass, test_calc], ignore_index=True)

    dicom_info_df = pd.read_csv(CSV_ROOT / "dicom_info.csv")

    logger.info(f"Train cases: {len(train_df)} (Mass: {len(train_mass)}, Calc: {len(train_calc)})")
    logger.info(f"Test cases: {len(test_df)} (Mass: {len(test_mass)}, Calc: {len(test_calc)})")
    logger.info(f"Train patients: {train_df['patient_id'].nunique()}")
    logger.info(f"Test patients: {test_df['patient_id'].nunique()}")

    return train_df, test_df, dicom_info_df


def main():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of training patients for validation (default: 0.1)"
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    args = parser.parse_args()

    logger.info("Starting CBIS-DDSM dataset preparation...")
    logger.info(f"Target image size: {TARGET_SIZE}")
    logger.info(f"Using official test split and creating validation split with ratio: {args.val_ratio}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, test_df, dicom_info_df = load_and_combine_data()
    print("Train data:")
    print(train_df)
    print("\nTest data:")
    print(test_df)
    print("\nDICOM info:")
    print(dicom_info_df)


if __name__ == "__main__":
    main()
