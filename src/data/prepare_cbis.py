"""
CBIS-DDSM dataset preparation script

This script prepares the CBIS-DDSM dataset for training by:
1. Loading mass and calcification case descriptions
2. Extracting ROI crops from the JPEG files
3. Resizing images to a fixed resolution (256x256)
4. Splitting data at patient level into train/val/test (70/10/20)
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

def main():
    parser = argparse.ArgumentParser(description="Prepare CBIS-DDSM dataset")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Proportion of patients for training (default: 0.7)"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Proportion of patients for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Proportion of patients for test (default: 0.2)"
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
    logger.info(f"Split ratios - Train: {args.train_ratio}, Val: {args.val_ratio}, Test: {args.test_ratio}")


if __name__ == "__main__":
    main()
