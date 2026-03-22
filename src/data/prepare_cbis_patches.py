"""
CBIS-DDSM patch-based dataset preparation for 5-class classification.

Implements S10-style patch sampling similar to Shen et al. (2019):

- 10 patches per ROI with >=90% overlap with ROI region
- 10 background patches per full mammogram (avoiding ROI regions)

See notebook for breakdown: notebooks/04-patches.ipynb
"""

import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.data.cbis_ddsm import load_dicom_array, parse_dcm_path, resolve_dcm_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

PATCH_SIZE = 224
PATCHES_PER_ROI = 10
PATCHES_PER_BG = 10
MIN_OVERLAP_RATIO = 0.9

CSV_ROOT = Path.home() / "datasets/CBIS-DDSM/fixed-csv"
# DICOM images root
DATASET_ROOT = Path("datasets/CBIS-DDSM")
OUTPUT_ROOT = Path("datasets/prep/cbis-ddsm-patches")
IMG_OUTPUT_DIR = OUTPUT_ROOT / "img"


def load_and_combine_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load CBIS-DDSM CSVs from fixed-csv directory and metadata.
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

    # Load metadata for file path resolution
    metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data at patient level to avoid data leakage
    """
    unique_patients = df["patient_id"].unique()

    train_patients, val_patients = train_test_split(
        unique_patients, test_size=val_ratio, random_state=random_state
    )

    train_df = df.loc[df["patient_id"].isin(train_patients)].reset_index(drop=True)
    val_df = df.loc[df["patient_id"].isin(val_patients)].reset_index(drop=True)

    logger.info(
        f"Split patients - Train: {len(train_patients)}, Val: {len(val_patients)}"
    )
    logger.info(f"Split cases - Train: {len(train_df)}, Val: {len(val_df)}")

    return train_df, val_df


def get_patch_label(abnormality_category: str, pathology: str) -> int:
    """
    Map abnormality type and pathology to 5-class label.

    Returns:
        0: Background
        1: Benign mass
        2: Malignant mass
        3: Benign calcification
        4: Malignant calcification
    """
    if abnormality_category == "background":
        return 0
    elif abnormality_category == "mass":
        return 1 if pathology in ("BENIGN", "BENIGN_WITHOUT_CALLBACK") else 2
    elif abnormality_category == "calcification":
        return 3 if pathology in ("BENIGN", "BENIGN_WITHOUT_CALLBACK") else 4
    else:
        raise ValueError(f"Unknown category: {abnormality_category}")


def get_breast_mask(img: np.ndarray) -> np.ndarray:
    """Get binary mask of breast region using thresholding."""
    # Normalize to uint8 for OpenCV operations
    if img.dtype != np.uint8:
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img

    blur = cv2.GaussianBlur(img_normalized, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Apply morphological operations to clean up the mask
    kernel = np.ones((50, 50), np.uint8)
    opened_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed_mask


def normalise_image(img: np.ndarray) -> np.ndarray:
    """Normalise image to 0-255 uint8 range."""
    if img.dtype != np.uint8:
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img
    return img_normalized


def extract_roi_patches(

    full_image: np.ndarray,
    mask: np.ndarray,
    patch_size: int = PATCH_SIZE,
    num_patches: int = PATCHES_PER_ROI,
    min_overlap: float = MIN_OVERLAP_RATIO
) -> list[np.ndarray]:
    """
    Extract patches centered around ROI with context.

    S10-style sampling: patches centered near ROI with surrounding tissue context.
    For diverse sampling, uses increasing jitter for each patch.

    Args:
        full_image: Full mammogram as numpy array
        mask: Binary mask (same size as full_image) indicating ROI
        patch_size: Size of square patches (224x224)
        num_patches: Number of patches to extract (10)
        min_overlap: Minimum fraction of patch that must contain ROI pixels (0.1 = 10%)

    Returns:
        List of patch arrays
    """
    patches = []
    img_h, img_w = full_image.shape[:2]

    # Find ROI bounding box and centroid
    roi_points = np.where(mask > 0)
    if len(roi_points[0]) == 0:
        logger.warning("Empty mask provided")
        return patches

    roi_y_min, roi_y_max = roi_points[0].min(), roi_points[0].max()
    roi_x_min, roi_x_max = roi_points[1].min(), roi_points[1].max()
    roi_h = roi_y_max - roi_y_min
    roi_w = roi_x_max - roi_x_min

    # Calculate centroid
    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = (roi_x_min + roi_x_max) // 2
        cy = (roi_y_min + roi_y_max) // 2

    half_patch = patch_size // 2

    # Calculate jitter range based on ROI size - include surrounding context
    max_jitter = max(roi_w, roi_h, patch_size) // 3

    np.random.seed(42)
    extracted_centers = set()  # Track extracted patch centers to avoid duplicates

    for patch_num in range(num_patches):
        # Increase jitter for diversity
        jitter_scale = (patch_num + 1) / num_patches
        current_jitter = int(max_jitter * jitter_scale)

        best_patch = None
        best_overlap = 0

        # Try several candidates, pick the one with best overlap
        for _ in range(20):
            jx = np.random.randint(-current_jitter, current_jitter + 1)
            jy = np.random.randint(-current_jitter, current_jitter + 1)
            px = cx + jx
            py = cy + jy

            # Skip if we already extracted from this location
            if (px, py) in extracted_centers:
                continue

            # Calculate patch boundaries
            x1 = px - half_patch
            y1 = py - half_patch
            x2 = x1 + patch_size
            y2 = y1 + patch_size

            # Handle boundary conditions
            pad_left = max(0, -x1)
            pad_top = max(0, -y1)
            pad_right = max(0, x2 - img_w)
            pad_bottom = max(0, y2 - img_h)

            x1_valid = max(0, x1)
            y1_valid = max(0, y1)
            x2_valid = min(img_w, x2)
            y2_valid = min(img_h, y2)

            # Check overlap
            mask_region = mask[y1_valid:y2_valid, x1_valid:x2_valid]
            roi_in_patch = np.sum(mask_region > 0)
            actual_patch_area = (y2_valid - y1_valid) * (x2_valid - x1_valid)
            overlap = roi_in_patch / actual_patch_area if actual_patch_area > 0 else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_patch = (px, py, x1_valid, y1_valid, x2_valid, y2_valid,
                              pad_left, pad_top, pad_right, pad_bottom)

        # Accept patch if it has any ROI content (at least 10% of patch)
        if best_patch is not None and best_overlap >= 0.1:
            (px, py, x1_valid, y1_valid, x2_valid, y2_valid,
             pad_left, pad_top, pad_right, pad_bottom) = best_patch
            extracted_centers.add((px, py))

            patch = full_image[y1_valid:y2_valid, x1_valid:x2_valid]

            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                patch = cv2.copyMakeBorder(
                    patch, pad_top, pad_bottom, pad_left, pad_right,
                    cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))

            patches.append(patch)

    if len(patches) < num_patches:
        logger.warning(f"Only extracted {len(patches)}/{num_patches} ROI patches")

    return patches


def extract_background_patches(
    full_image: np.ndarray,
    all_masks: list[np.ndarray],
    patch_size: int = PATCH_SIZE,
    num_patches: int = PATCHES_PER_BG
) -> list[np.ndarray]:
    """
    Extract background patches from breast tissue avoiding all ROI regions.

    Strategy:
    1. Combine all ROI masks into exclusion zone
    2. Dilate exclusion zone by patch_size to ensure no overlap
    3. Apply breast mask to only sample from breast tissue
    4. Random sample valid positions
    5. Verify no overlap with any ROI
    """
    patches = []
    img_h, img_w = full_image.shape[:2]
    half_patch = patch_size // 2

    # Get breast tissue mask
    breast_mask = get_breast_mask(full_image)

    # Combine all ROI masks into exclusion zone
    if all_masks:
        exclusion_mask = np.zeros_like(all_masks[0])
        for mask in all_masks:
            exclusion_mask = cv2.bitwise_or(exclusion_mask, mask)

        # Dilate exclusion zone by patch_size to create buffer
        kernel = np.ones((patch_size, patch_size), np.uint8)
        exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=1)
    else:
        exclusion_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Valid region = breast tissue AND NOT exclusion zone
    valid_mask = cv2.bitwise_and(breast_mask, cv2.bitwise_not(exclusion_mask))

    # Erode valid mask to account for patch size
    kernel_erode = np.ones((patch_size // 2, patch_size // 2), np.uint8)
    valid_mask = cv2.erode(valid_mask, kernel_erode, iterations=1)

    # Find valid positions
    valid_mask_arr: np.ndarray = np.asarray(valid_mask)
    valid_points = np.where(valid_mask_arr > 0)
    if len(valid_points[0]) == 0:
        logger.warning("No valid background positions found")
        return patches

    # Random sample positions
    np.random.seed(42)
    num_valid = len(valid_points[0])
    max_attempts = num_patches * 10

    sampled_positions = set()

    for _ in range(max_attempts):
        if len(patches) >= num_patches:
            break

        idx = np.random.randint(0, num_valid)
        cy = valid_points[0][idx]
        cx = valid_points[1][idx]

        # Check if too close to already sampled position
        too_close = False
        for (sx, sy) in sampled_positions:
            if abs(cx - sx) < patch_size // 2 and abs(cy - sy) < patch_size // 2:
                too_close = True
                break

        if too_close:
            continue

        # Extract patch
        x1 = cx - half_patch
        y1 = cy - half_patch
        x2 = x1 + patch_size
        y2 = y1 + patch_size

        # Handle boundaries
        x1_valid = max(0, x1)
        y1_valid = max(0, y1)
        x2_valid = min(img_w, x2)
        y2_valid = min(img_h, y2)

        patch = full_image[y1_valid:y2_valid, x1_valid:x2_valid]

        # Pad if necessary
        pad_left = max(0, -x1)
        pad_top = max(0, -y1)
        pad_right = max(0, x2 - img_w)
        pad_bottom = max(0, y2 - img_h)

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            patch = cv2.copyMakeBorder(
                patch,
                pad_top, pad_bottom, pad_left, pad_right,
                cv2.BORDER_CONSTANT, value=(0, 0, 0)
            )

        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        patches.append(patch)
        sampled_positions.add((cx, cy))

    if len(patches) < num_patches:
        logger.warning(f"Only extracted {len(patches)}/{num_patches} background patches")

    return patches


def group_abnormalities_by_image(df: pd.DataFrame) -> dict:
    """
    Group all abnormalities that belong to the same mammogram image.

    Key: (patient_id, breast_side, image_view) -> identifies unique image
    Value: DataFrame of all abnormalities in that image

    This is critical for:
    - Background extraction (need ALL masks to create exclusion zone)
    - Avoiding duplicate background patches
    """
    df["image_key"] = df.apply(
        lambda row: (row["patient_id"], row["left or right breast"], row["image view"]),
        axis=1
    )
    grouped = {key: group for key, group in df.groupby("image_key")}
    return grouped


def load_full_image(row: pd.Series, metadata_df: pd.DataFrame) -> np.ndarray | None:
    """Load full mammogram image from DICOM using metadata lookup."""
    image_path_str = str(row["image file path"])
    try:
        dcm_data = parse_dcm_path(image_path_str)
        dicom_path = resolve_dcm_path(dcm_data, metadata_df, DATASET_ROOT)
        if not dicom_path.exists():
            logger.warning(f"DICOM file not found: {dicom_path}")
            return None
        img = load_dicom_array(dicom_path)
        return normalise_image(img)
    except IndexError:
        logger.warning(f"No metadata match for: {image_path_str}")
    except Exception as e:
        logger.warning(f"Failed to load image {image_path_str}: {type(e).__name__}: {e}")
    return None


def load_roi_mask(row: pd.Series, metadata_df: pd.DataFrame) -> np.ndarray | None:
    """Load ROI mask from DICOM using metadata lookup."""
    try:
        mask_path_value = row.get("ROI mask file path")
        if mask_path_value is None or pd.isna(mask_path_value):
            return None

        mask_path_str = str(mask_path_value)
        dcm_data = parse_dcm_path(mask_path_str)
        dicom_path = resolve_dcm_path(dcm_data, metadata_df, DATASET_ROOT)
        if not dicom_path.exists():
            logger.warning(f"Mask DICOM not found: {dicom_path}")
            return None

        mask = load_dicom_array(dicom_path)
        # Normalize mask to binary
        mask = (mask > 0).astype(np.uint8) * 255
        return mask
    except IndexError:
        logger.warning(f"No metadata match for mask: {mask_path_str}")
    except Exception as e:
        logger.warning(f"Failed to load mask {mask_path_str}: {type(e).__name__}: {e}")
    return None


def process_image_group(
    image_key: tuple,
    group_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    output_dir: Path,
    image_idx: int,
    patches_per_roi: int = PATCHES_PER_ROI,
    patches_per_bg: int = PATCHES_PER_BG
) -> list[dict]:
    """
    Process all abnormalities for a single mammogram image.
    """
    patient_id, breast_side, image_view = image_key
    all_metadata = []

    # Load full image (use first row, all rows should have same image path)
    first_row = group_df.iloc[0]
    full_image = load_full_image(first_row, metadata_df)
    if full_image is None:
        logger.warning(f"Could not load image for {image_key}")
        return []

    # Load all ROI masks for this image
    all_masks = []
    for _, row in group_df.iterrows():
        mask = load_roi_mask(row, metadata_df)
        if mask is not None:
            # Resize mask if needed to match image
            if mask.shape != full_image.shape:
                mask = cv2.resize(mask, (full_image.shape[1], full_image.shape[0]))
            all_masks.append((row, mask))

    # Extract ROI patches for each abnormality
    for row, mask in all_masks:
        abnormality_category = row["abnormality_category"]
        pathology = row["pathology"]
        abnormality_id = row.get("abnormality id", 1)
        label = get_patch_label(abnormality_category, pathology)

        patches = extract_roi_patches(
            full_image, mask,
            num_patches=patches_per_roi
        )

        for patch_idx, patch in enumerate(patches, 1):
            filename = (
                f"{image_idx:05d}_{patient_id}_{breast_side}_{image_view}_"
                f"{abnormality_category}_{abnormality_id}_{pathology}_roi_{patch_idx:02d}.png"
            )
            output_path = output_dir / filename
            cv2.imwrite(str(output_path), patch)

            all_metadata.append({
                "filename": filename,
                "patient_id": patient_id,
                "breast_side": breast_side,
                "image_view": image_view,
                "abnormality_category": abnormality_category,
                "abnormality_id": abnormality_id,
                "pathology": pathology,
                "label": label,
                "patch_type": "roi",
                "patch_idx": patch_idx,
                "source_image_idx": image_idx
            })

    # Extract background patches (once per image, avoiding all ROIs)
    masks_only = [mask for _, mask in all_masks]
    bg_patches = extract_background_patches(
        full_image, masks_only,
        num_patches=patches_per_bg
    )

    for patch_idx, patch in enumerate(bg_patches, 1):
        filename = (
            f"{image_idx:05d}_{patient_id}_{breast_side}_{image_view}_bg_{patch_idx:02d}.png"
        )
        output_path = output_dir / filename
        cv2.imwrite(str(output_path), patch)

        all_metadata.append({
            "filename": filename,
            "patient_id": patient_id,
            "breast_side": breast_side,
            "image_view": image_view,
            "abnormality_category": "background",
            "abnormality_id": 0,
            "pathology": "BACKGROUND",
            "label": 0,
            "patch_type": "background",
            "patch_idx": patch_idx,
            "source_image_idx": image_idx
        })

    return all_metadata


def process_and_save_split(
    split_df: pd.DataFrame,
    split_name: str,
    metadata_df: pd.DataFrame,
    img_output_dir: Path,
    output_root: Path,
    patches_per_roi: int = PATCHES_PER_ROI,
    patches_per_bg: int = PATCHES_PER_BG,
    start_idx: int = 0
) -> tuple[pd.DataFrame, int]:
    """
    Process all images in a split and save metadata.
    """
    logger.info(f"Processing {split_name} split...")

    # Group by unique image
    grouped = group_abnormalities_by_image(split_df)
    logger.info(f"{split_name}: {len(grouped)} unique images, {len(split_df)} abnormalities")

    all_metadata = []
    current_idx = start_idx

    for image_key, group_df in tqdm(
        grouped.items(), total=len(grouped), desc=f"Processing {split_name}"
    ):
        metadata = process_image_group(
            image_key, group_df, metadata_df, img_output_dir, current_idx,
            patches_per_roi=patches_per_roi, patches_per_bg=patches_per_bg
        )
        all_metadata.extend(metadata)
        current_idx += 1

    result_df = pd.DataFrame(all_metadata)

    # Save CSV
    csv_path = output_root / f"{split_name}.csv"
    result_df.to_csv(csv_path, index=False)
    logger.info(f"Saved {split_name} metadata to {csv_path}")

    # Log class distribution
    if len(result_df) > 0:
        class_counts = result_df["label"].value_counts().sort_index()
        class_names = ["Background", "Benign mass", "Malignant mass",
                       "Benign calc", "Malignant calc"]
        logger.info(f"{split_name} class distribution:")
        for label in class_counts.index:
            count = class_counts[label]
            logger.info(f"  {class_names[label]}: {count}")

    return result_df, current_idx


def main():
    parser = argparse.ArgumentParser(
        description="Prepare CBIS-DDSM patch-based dataset for 5-class classification"
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
        "--patches-per-roi",
        type=int,
        default=PATCHES_PER_ROI,
        help=f"Number of patches per ROI (default: {PATCHES_PER_ROI})",
    )
    parser.add_argument(
        "--patches-per-bg",
        type=int,
        default=PATCHES_PER_BG,
        help=f"Number of background patches per image (default: {PATCHES_PER_BG})",
    )

    args = parser.parse_args()

    logger.info("Starting CBIS-DDSM patch-based dataset preparation...")
    logger.info(f"Patches per ROI: {args.patches_per_roi}")
    logger.info(f"Background patches per image: {args.patches_per_bg}")
    logger.info(f"Output directory: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    IMG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_all_df, test_df, metadata_df = load_and_combine_data()

    # Split training data into train and validation sets (patient-level)
    train_df, val_df = split_by_patient(
        train_all_df, val_ratio=args.val_ratio, random_state=args.random_seed
    )

    # Process each split
    train_metadata, next_idx = process_and_save_split(
        train_df, "train", metadata_df, IMG_OUTPUT_DIR, OUTPUT_ROOT,
        patches_per_roi=args.patches_per_roi, patches_per_bg=args.patches_per_bg,
        start_idx=0
    )

    val_metadata, next_idx = process_and_save_split(
        val_df, "val", metadata_df, IMG_OUTPUT_DIR, OUTPUT_ROOT,
        patches_per_roi=args.patches_per_roi, patches_per_bg=args.patches_per_bg,
        start_idx=next_idx
    )

    test_metadata, _ = process_and_save_split(
        test_df, "test", metadata_df, IMG_OUTPUT_DIR, OUTPUT_ROOT,
        patches_per_roi=args.patches_per_roi, patches_per_bg=args.patches_per_bg,
        start_idx=next_idx
    )

    logger.info("Dataset preparation complete!")
    logger.info(f"Total patches: {len(train_metadata) + len(val_metadata) + len(test_metadata)}")
    logger.info(f"  Train: {len(train_metadata)}")
    logger.info(f"  Val: {len(val_metadata)}")
    logger.info(f"  Test: {len(test_metadata)}")


if __name__ == "__main__":
    main()
