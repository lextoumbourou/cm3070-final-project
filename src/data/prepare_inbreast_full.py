import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

TARGET_SIZE = 512

RAW_DATA_ROOT = Path("datasets/INbreast Release 1.0")
DICOM_DIR = RAW_DATA_ROOT / "AllDICOMs"
OUTPUT_ROOT = Path("datasets/prep/inbreast")
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
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_shuffled = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    n_total = len(df_shuffled)
    n_test = int(n_total * test_ratio)
    n_val = int(n_total * val_ratio)

    test_df = df_shuffled[:n_test].reset_index(drop=True)
    val_df = df_shuffled[n_test:n_test + n_val].reset_index(drop=True)
    train_df = df_shuffled[n_test + n_val:].reset_index(drop=True)

    logger.info(f"Split cases - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    return train_df, val_df, test_df


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


def crop_coords(img_array):
    if img_array.max() > 255:
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img_array = (img_array * 255).astype(np.uint8)
    blur = cv2.GaussianBlur(img_array, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    morph_img = apply_morphological_transforms(breast_mask)
    return get_contours_from_mask(morph_img)


def truncation_normalisation(img):
    Pmin = np.percentile(img[img != 0], 5)
    Pmax = np.percentile(img[img != 0], 99)
    truncated = np.clip(img, Pmin, Pmax)
    normalized = (truncated - Pmin) / (Pmax - Pmin)
    normalized[img == 0] = 0
    return normalized


def clahe(img, clip):
    clahe_obj = cv2.createCLAHE(clipLimit=clip)
    cl = clahe_obj.apply(np.array(img * 255, dtype=np.uint8))
    return cl


def preprocess_mammogram(img_array, target_size=TARGET_SIZE):
    if img_array.max() > 255:
        img_array = (img_array - img_array.min()) / (img_array.max() - img_array.min())
        img_array = (img_array * 255).astype(np.uint8)

    x, y, w, h = crop_coords(img_array)
    img_cropped = img_array[y:y+h, x:x+w]
    img_normalized = truncation_normalisation(img_cropped)
    cl1 = clahe(img_normalized, 1.0)
    cl2 = clahe(img_normalized, 2.0)
    img_final = cv2.merge((np.array(img_normalized * 255, dtype=np.uint8), cl1, cl2))
    img_final = cv2.resize(img_final, (target_size, target_size))
    return img_final


def process_case(row: pd.Series, output_dir: Path, case_idx: int) -> dict:
    file_name = str(row["File Name"])
    dicom_files = list(DICOM_DIR.rglob(f"{file_name}*.dcm"))

    if not dicom_files:
        logger.warning(f"DICOM not found for: {file_name}")
        return None

    try:
        dicom_data = pydicom.dcmread(dicom_files[0])
        img_array = dicom_data.pixel_array
        img_processed = preprocess_mammogram(img_array)
    except Exception as e:
        logger.warning(f"Failed to preprocess {file_name}: {e}")
        return None

    patient_id = row["Patient ID"]
    laterality = row["Laterality"]
    view = row["View"]
    pathology = row["pathology"]
    birads = row["Bi-Rads"]

    filename = f"{case_idx:05d}_{patient_id}_{laterality}_{view}_{pathology}.png"
    output_path = output_dir / filename

    cv2.imwrite(str(output_path), cv2.cvtColor(img_processed, cv2.COLOR_RGB2BGR))

    label = 1 if pathology == "MALIGNANT" else 0

    return {
        "filename": filename,
        "patient_id": patient_id,
        "laterality": laterality,
        "view": view,
        "birads": birads,
        "acr": row.get("ACR", None),
        "pathology": pathology,
        "label": label,
    }


def process_and_save_split(
    split_df: pd.DataFrame,
    split_name: str,
    img_output_dir: Path,
) -> pd.DataFrame:
    logger.info(f"Processing {split_name} split...")

    processed_cases = []

    for idx, row in tqdm(
        split_df.iterrows(), total=len(split_df), desc=f"Processing {split_name}"
    ):
        metadata = process_case(row, img_output_dir, idx)
        if metadata is not None:
            processed_cases.append(metadata)

    metadata_df = pd.DataFrame(processed_cases)

    csv_path = OUTPUT_ROOT / f"{split_name}.csv"
    metadata_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    logger.info(f"{split_name} - Total cases: {len(metadata_df)}")
    logger.info(f"{split_name} - Benign: {(metadata_df['label'] == 0).sum()}")
    logger.info(f"{split_name} - Malignant: {(metadata_df['label'] == 1).sum()}")

    return metadata_df


def main():
    parser = argparse.ArgumentParser(description="Prepare INbreast dataset")
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Proportion of patients for test set (default: 0.15)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Proportion of patients for validation set (default: 0.15)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    logger.info("Starting INbreast dataset preparation...")
    logger.info(f"Target image size: {TARGET_SIZE}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_metadata()

    train_df, val_df, test_df = split_data(
        df,
        test_ratio=args.test_ratio,
        val_ratio=args.val_ratio,
        random_state=args.random_seed,
    )

    process_and_save_split(train_df, "train", IMG_OUTPUT_DIR)
    process_and_save_split(val_df, "val", IMG_OUTPUT_DIR)
    process_and_save_split(test_df, "test", IMG_OUTPUT_DIR)

    logger.info("INbreast dataset preparation complete!")
    logger.info(f"Processed images saved to: {IMG_OUTPUT_DIR}")
    logger.info(f"Metadata CSVs saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
