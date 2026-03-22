# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Patch-Based Preprocessing Pipeline
#
# This notebook demonstrates the patch-based preprocessing pipeline for 5-class mammography classification,
# following the approach from Shen et al. (2019).
#
# ## Patch Sampling Strategy:
#
# - 10 ROI patches per abnormality (with surrounding tissue context)
# - 10 background patches per image (avoiding ROI regions)

# %%
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom
from pydantic import BaseModel

# Add src to path for imports
sys.path.insert(0, str(Path("..").resolve()))
from src.trainer_multiclass import get_train_transform, get_val_transform

# %% [markdown]
# ## Setup and Helper Functions
#
# Code is mostly reused from the `cbis-ddsm` notebook.
#
# todo: turn this into library code.

# %%
DATASET_ROOT = Path("../datasets/CBIS-DDSM")
CSV_ROOT = Path.home() / "datasets/CBIS-DDSM/fixed-csv"

CLASS_NAMES = ["Background", "Benign mass", "Malignant mass", "Benign calc", "Malignant calc"]

class DCMData(BaseModel):
    subject_id: str
    study_uid: str
    series_uid: str
    dcm_file: str


def get_file_data_from_dcm(dcm_path: str) -> DCMData:
    """Parse DICOM file path to extract components."""
    data = str(dcm_path).strip().split("/")
    dcm_file = data[-1].strip().split(".")[0]
    return DCMData(subject_id=data[0], study_uid=data[1], series_uid=data[2], dcm_file=dcm_file)


def get_filepath_from_dcm_data(dcm_data: DCMData, metadata_df: pd.DataFrame) -> Path:
    """Get the full file path from parsed DCM data using metadata lookup."""
    meta = metadata_df[
        (metadata_df["Subject ID"] == dcm_data.subject_id) &
        (metadata_df["Series UID"] == dcm_data.series_uid) &
        (metadata_df["Study UID"] == dcm_data.study_uid)
    ].iloc[0]
    file_location = meta["File Location"]
    return DATASET_ROOT / Path(file_location) / (dcm_data.dcm_file + ".dcm")


def dicom_to_array(file_path: Path) -> np.ndarray:
    """Load a DICOM file and return as numpy array."""
    ds = pydicom.dcmread(file_path)
    return ds.pixel_array


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 0-255 uint8 range."""
    if img.dtype != np.uint8:
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img
    return img_normalized


def get_patch_label(abnormality_category: str, pathology: str) -> int:
    """Map abnormality type and pathology to 5-class label."""
    if abnormality_category == "background":
        return 0
    elif abnormality_category == "mass":
        return 1 if pathology in ("BENIGN", "BENIGN_WITHOUT_CALLBACK") else 2
    elif abnormality_category == "calcification":
        return 3 if pathology in ("BENIGN", "BENIGN_WITHOUT_CALLBACK") else 4
    else:
        raise ValueError(f"Unknown category: {abnormality_category}")


# %% [markdown]
# ## Load Sample Data

# %%
# Load metadata
train_mass_df = pd.read_csv(CSV_ROOT / "mass_case_description_train_set.csv")
train_calc_df = pd.read_csv(CSV_ROOT / "calc_case_description_train_set.csv")
metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")

train_mass_df["abnormality_category"] = "mass"
train_calc_df["abnormality_category"] = "calcification"

print(f"Mass cases: {len(train_mass_df)}")
print(f"Calcification cases: {len(train_calc_df)}")

# %%
# Select a sample mass case
sample_row = train_mass_df[train_mass_df["patient_id"] == "P_00001"].iloc[0]
print(f"Patient: {sample_row['patient_id']}")
print(f"View: {sample_row['image view']}")
print(f"Pathology: {sample_row['pathology']}")
print(f"Abnormality: {sample_row['abnormality_category']}")

# %% [markdown]
# ## Load Full Mammogram and ROI Mask

# %%
# Load the full mammogram image
img_dcm = get_file_data_from_dcm(sample_row["image file path"])
img_path = get_filepath_from_dcm_data(img_dcm, metadata_df)
full_image = normalize_image(dicom_to_array(img_path))

# Load the ROI mask
mask_dcm = get_file_data_from_dcm(sample_row["ROI mask file path"])
mask_path = get_filepath_from_dcm_data(mask_dcm, metadata_df)
roi_mask = dicom_to_array(mask_path)
roi_mask = (roi_mask > 0).astype(np.uint8) * 255

print(f"Full image shape: {full_image.shape}")
print(f"ROI mask shape: {roi_mask.shape}")

# %%
# Visualize the mammogram with ROI overlay
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(full_image, cmap='gray')
axes[0].set_title("Full Mammogram")
axes[0].axis('off')

axes[1].imshow(roi_mask, cmap='gray')
axes[1].set_title("ROI Mask")
axes[1].axis('off')

# Overlay
axes[2].imshow(full_image, cmap='gray')
axes[2].imshow(roi_mask, cmap='Reds', alpha=0.4)
axes[2].set_title("Mammogram with ROI Overlay")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## ROI Patch Extraction
#
# Extract 10 patches centered around the ROI region. Each patch includes surrounding tissue context,
# which is important for whole-image classification performance (as noted in Shen et al.).
#
# Walking through the algorithm.
#
# We're aiming to get patch sizes of size 224, around the region of interest.

# %%
PATCH_SIZE = 224

# %% [markdown]
# Firstly, we want to extract the centroid of the image:

# %%
roi_points = np.where(roi_mask > 0)

roi_y_min, roi_y_max = roi_points[0].min(), roi_points[0].max()
roi_x_min, roi_x_max = roi_points[1].min(), roi_points[1].max()

# %%
roi_h = roi_y_max - roi_y_min
roi_w = roi_x_max - roi_x_min

roi_h, roi_w

# %% [markdown]
# We can use `cv2.moments` to find the center of a blob.
#
# Source: https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python

# %%
moments = cv2.moments(roi_mask)
cx = int(moments["m10"] / moments["m00"])
cy = int(moments["m01"] / moments["m00"])

# %% [markdown]
# We set the max jitter to define how far around the ROI we are allowed to go.

# %%
half_patch = PATCH_SIZE // 2
max_jitter = max(roi_w, roi_h, PATCH_SIZE) // 3
max_jitter

# %% [markdown]
# Now we set up the random seed for reproducibility and initialise a set to track extracted patch centres (to avoid duplicates):

# %% [markdown]
# Get the image dimensions for boundary checking:

# %% [markdown]
# ### Main Loop: Progressive Jitter Sampling
#
# For each of the 10 patches, we use progressive jitter scaling. Earlier patches stay closer to the centroid, while later patches can wander further. This ensures diverse coverage of the ROI region:

# %%
num_patches = 10

for patch_num in range(num_patches):
    jitter_scale = (patch_num + 1) / num_patches
    current_jitter = int(max_jitter * jitter_scale)
    print(f"Patch {patch_num}: jitter_scale={jitter_scale:.1f}, current_jitter=±{current_jitter}px")

# %% [markdown]
# ### Inner Loop: Best Overlap Selection
#
# For each patch, we try up to 20 random candidate positions within the current jitter radius. We select the candidate with the highest ROI overlap (at least 10% of the patch must contain ROI pixels):

# %%
# Demonstrate overlap calculation for a single candidate
patch_num = 0
current_jitter = int(max_jitter * 0.1)  # First patch has smallest jitter

# Generate a random offset within jitter range
np.random.seed(42)
jx = np.random.randint(-current_jitter, current_jitter + 1)
jy = np.random.randint(-current_jitter, current_jitter + 1)
px, py = cx + jx, cy + jy

print(f"Centroid: ({cx}, {cy})")
print(f"Jitter applied: ({jx}, {jy})")
print(f"Candidate centre: ({px}, {py})")

# Calculate patch boundaries
x1, y1 = px - half_patch, py - half_patch
x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE
print(f"\nPatch boundaries: x=[{x1}, {x2}], y=[{y1}, {y2}]")

# %% [markdown]
# Clamp boundaries to valid image region and calculate ROI overlap:

# %%

# %%
img_w, img_h = full_image.shape

# %%
# Clamp to image boundaries
x1_valid = max(0, x1)
y1_valid = max(0, y1)
x2_valid = min(img_w, x2)
y2_valid = min(img_h, y2)

print(f"Clamped boundaries: x=[{x1_valid}, {x2_valid}], y=[{y1_valid}, {y2_valid}]")

# Extract the mask region and calculate overlap
mask_region = roi_mask[y1_valid:y2_valid, x1_valid:x2_valid]
roi_pixels_in_patch = np.sum(mask_region > 0)
actual_patch_area = (y2_valid - y1_valid) * (x2_valid - x1_valid)
overlap = roi_pixels_in_patch / actual_patch_area if actual_patch_area > 0 else 0

print(f"\nROI pixels in patch: {roi_pixels_in_patch}")
print(f"Patch area: {actual_patch_area}")
print(f"Overlap ratio: {overlap:.2%}")

# %% [markdown]
# Visualise this candidate patch location:

# %%
import matplotlib.patches as mpatches

# Zoom into the ROI region with some padding
padding = 100
zoom_x1 = max(0, roi_x_min - padding)
zoom_y1 = max(0, roi_y_min - padding)
zoom_x2 = min(img_w, roi_x_max + padding + PATCH_SIZE)
zoom_y2 = min(img_h, roi_y_max + padding + PATCH_SIZE)

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(full_image[zoom_y1:zoom_y2, zoom_x1:zoom_x2], cmap='gray')
ax.imshow(roi_mask[zoom_y1:zoom_y2, zoom_x1:zoom_x2], cmap='Reds', alpha=0.4)

# Draw the candidate patch (adjusted for zoom)
rect = mpatches.Rectangle(
    (x1_valid - zoom_x1, y1_valid - zoom_y1), 
    PATCH_SIZE, PATCH_SIZE,
    linewidth=2, edgecolor='lime', facecolor='none'
)
ax.add_patch(rect)

# Mark centroid and candidate centre
ax.plot(cx - zoom_x1, cy - zoom_y1, 'b+', markersize=15, markeredgewidth=2, label='ROI centroid')
ax.plot(px - zoom_x1, py - zoom_y1, 'go', markersize=8, label='Patch centre')

ax.legend(loc='upper right')
ax.set_title(f"Candidate Patch (overlap: {overlap:.1%})")
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Edge Handling: Padding
#
# If the patch extends beyond the image boundary, we need to pad with zeros. This happens when the ROI is near the edge of the mammogram:

# %%
# Calculate required padding (for our candidate, this should be zero since ROI is not near edge)
pad_left = max(0, -(px - half_patch))
pad_top = max(0, -(py - half_patch))
pad_right = max(0, (px + half_patch) - img_w)
pad_bottom = max(0, (py + half_patch) - img_h)

print(f"Padding required: left={pad_left}, top={pad_top}, right={pad_right}, bottom={pad_bottom}")

if pad_left == 0 and pad_top == 0 and pad_right == 0 and pad_bottom == 0:
    print("No padding needed - patch is fully within image bounds")

# %% [markdown]
# ### Final Extraction
#
# Extract the patch from the full image, apply padding if needed, and resize to ensure consistent 224×224 output:

# %%
# Extract the patch
patch = full_image[y1_valid:y2_valid, x1_valid:x2_valid].copy()
print(f"Extracted patch shape: {patch.shape}")

# Apply padding if necessary
if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
    patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                               cv2.BORDER_CONSTANT, value=0)
    print(f"After padding: {patch.shape}")

# Resize if dimensions don't match (shouldn't happen if logic is correct)
if patch.shape[0] != PATCH_SIZE or patch.shape[1] != PATCH_SIZE:
    patch = cv2.resize(patch, (PATCH_SIZE, PATCH_SIZE))
    print(f"After resize: {patch.shape}")

print(f"\nFinal patch shape: {patch.shape}")

# %%
# Visualise the extracted patch
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(patch, cmap='gray')
axes[0].set_title(f"Extracted Patch ({PATCH_SIZE}×{PATCH_SIZE})")
axes[0].axis('off')

# Show corresponding mask region
mask_patch = roi_mask[y1_valid:y2_valid, x1_valid:x2_valid]
axes[1].imshow(patch, cmap='gray')
axes[1].imshow(mask_patch, cmap='Reds', alpha=0.5)
axes[1].set_title(f"Patch with ROI Overlay ({overlap:.1%} overlap)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# The full function wraps all the above steps:

# %%
PATCH_SIZE = 224


def extract_roi_patches(full_image, mask, patch_size=224, num_patches=10):
    """
    Extract patches from ROI region with surrounding context.

    Uses increasing jitter from centroid for diverse sampling.
    """
    patches_list = []
    img_h, img_w = full_image.shape[:2]

    # Find ROI centroid
    roi_points = np.where(mask > 0)
    if len(roi_points[0]) == 0:
        return []

    roi_y_min, roi_y_max = roi_points[0].min(), roi_points[0].max()
    roi_x_min, roi_x_max = roi_points[1].min(), roi_points[1].max()
    roi_h = roi_y_max - roi_y_min
    roi_w = roi_x_max - roi_x_min

    moments = cv2.moments(mask)
    if moments["m00"] > 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = (roi_x_min + roi_x_max) // 2
        cy = (roi_y_min + roi_y_max) // 2

    half_patch = patch_size // 2
    max_jitter = max(roi_w, roi_h, patch_size) // 3

    np.random.seed(42)
    extracted_centers = set()

    for patch_num in range(num_patches):
        jitter_scale = (patch_num + 1) / num_patches
        current_jitter = int(max_jitter * jitter_scale)

        best_patch = None
        best_overlap = 0
        best_coords = None

        for _ in range(20):
            jx = np.random.randint(-current_jitter, current_jitter + 1)
            jy = np.random.randint(-current_jitter, current_jitter + 1)
            px, py = cx + jx, cy + jy

            if (px, py) in extracted_centers:
                continue

            x1, y1 = px - half_patch, py - half_patch
            x2, y2 = x1 + patch_size, y1 + patch_size

            x1_valid = max(0, x1)
            y1_valid = max(0, y1)
            x2_valid = min(img_w, x2)
            y2_valid = min(img_h, y2)

            mask_region = mask[y1_valid:y2_valid, x1_valid:x2_valid]
            roi_in_patch = np.sum(mask_region > 0)
            actual_patch_area = (y2_valid - y1_valid) * (x2_valid - x1_valid)
            overlap = roi_in_patch / actual_patch_area if actual_patch_area > 0 else 0

            if overlap > best_overlap:
                best_overlap = overlap
                best_coords = (px, py, x1_valid, y1_valid, x2_valid, y2_valid)

        if best_coords is not None and best_overlap >= 0.1:
            px, py, x1_valid, y1_valid, x2_valid, y2_valid = best_coords
            extracted_centers.add((px, py))

            patch = full_image[y1_valid:y2_valid, x1_valid:x2_valid]

            # Pad if necessary
            pad_left = max(0, -(px - half_patch))
            pad_top = max(0, -(py - half_patch))
            pad_right = max(0, (px + half_patch) - img_w)
            pad_bottom = max(0, (py + half_patch) - img_h)

            if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
                patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                           cv2.BORDER_CONSTANT, value=0)

            if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
                patch = cv2.resize(patch, (patch_size, patch_size))

            patches_list.append((patch, (px, py)))

    return patches_list


# %%
# Extract ROI patches
roi_patches = extract_roi_patches(full_image, roi_mask, patch_size=PATCH_SIZE, num_patches=10)
print(f"Extracted {len(roi_patches)} ROI patches")

# %%
# Visualize ROI patches
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

label = get_patch_label(sample_row["abnormality_category"], sample_row["pathology"])
label_name = CLASS_NAMES[label]

for i, (patch, (px, py)) in enumerate(roi_patches):
    axes[i].imshow(patch, cmap='gray')
    axes[i].set_title(f"ROI Patch {i+1}\n{label_name}", fontsize=9)
    axes[i].axis('off')

plt.suptitle(f"ROI Patches (Class {label}: {label_name})", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize Patch Locations on Mammogram
#
# Show where the ROI patches were extracted from on the original mammogram.

# %%
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(full_image, cmap='gray')
ax.imshow(roi_mask, cmap='Reds', alpha=0.3)

# Draw patch locations
import matplotlib.patches as mpatches

colors = plt.cm.viridis(np.linspace(0, 1, len(roi_patches)))

for i, (patch, (px, py)) in enumerate(roi_patches):
    half = PATCH_SIZE // 2
    rect = mpatches.Rectangle((px - half, py - half), PATCH_SIZE, PATCH_SIZE,
                                linewidth=2, edgecolor=colors[i], facecolor='none')
    ax.add_patch(rect)
    ax.plot(px, py, 'o', color=colors[i], markersize=5)

ax.set_title("ROI Patch Locations on Mammogram")
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Background Patch Extraction
#
# Extract 10 background patches from breast tissue, avoiding the ROI regions.
# These patches serve as negative examples for the classifier.
#
# ### Step 1: Create Breast Tissue Mask
#
# First, we need to identify where the breast tissue is (vs. the black background of the mammogram). We use Otsu's thresholding followed by morphological operations to get a clean mask:

# %%
# Step 1a: Apply Gaussian blur to reduce noise
blur = cv2.GaussianBlur(full_image, (5, 5), 0)

# Step 1b: Otsu's thresholding to separate breast from background
_, breast_mask_raw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

print(f"Breast mask shape: {breast_mask_raw.shape}")
print(f"Unique values: {np.unique(breast_mask_raw)}")

# %% [markdown]
# As per the `03-preprocess` notebook, we use morphological operations to clean up the mask.

# %%
# Large kernel for significant morphological effect
kernel = np.ones((50, 50), np.uint8)

# Opening removes small bright spots (noise)
opened_mask = cv2.morphologyEx(breast_mask_raw, cv2.MORPH_OPEN, kernel)

# Closing fills small dark holes within the breast
breast_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

print(f"Kernel size: {kernel.shape}")

# %%
# Visualise the breast mask creation process
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

axes[0].imshow(full_image, cmap='gray')
axes[0].set_title("Original Mammogram")
axes[0].axis('off')

axes[1].imshow(breast_mask_raw, cmap='gray')
axes[1].set_title("After Otsu Threshold")
axes[1].axis('off')

axes[2].imshow(opened_mask, cmap='gray')
axes[2].set_title("After Opening")
axes[2].axis('off')

axes[3].imshow(breast_mask, cmap='gray')
axes[3].set_title("After Closing (Final)")
axes[3].axis('off')

plt.suptitle("Breast Tissue Mask Creation", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step 2: Create Exclusion Zone
#
# We need to avoid sampling patches that overlap with the ROI. We dilate the ROI mask by the patch size to create an exclusion zone:

# %%
# Dilate ROI mask to create exclusion zone
# Kernel size = patch size ensures no patch centre can produce a patch overlapping the ROI
exclusion_kernel = np.ones((PATCH_SIZE, PATCH_SIZE), np.uint8)
exclusion_mask = cv2.dilate(roi_mask, exclusion_kernel, iterations=1)

print(f"Original ROI mask pixels: {np.sum(roi_mask > 0)}")
print(f"Exclusion zone pixels: {np.sum(exclusion_mask > 0)}")
print(f"Expansion factor: {np.sum(exclusion_mask > 0) / np.sum(roi_mask > 0):.1f}x")

# %%
# Visualise ROI vs exclusion zone
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(roi_mask, cmap='gray')
axes[0].set_title("Original ROI Mask")
axes[0].axis('off')

axes[1].imshow(exclusion_mask, cmap='gray')
axes[1].set_title(f"Exclusion Zone (dilated by {PATCH_SIZE}px)")
axes[1].axis('off')

# Overlay showing the expansion
axes[2].imshow(full_image, cmap='gray')
axes[2].imshow(exclusion_mask, cmap='Reds', alpha=0.3)
axes[2].imshow(roi_mask, cmap='Blues', alpha=0.5)
axes[2].set_title("Exclusion Zone (red) vs ROI (blue)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step 3: Create Valid Sampling Region
#
# The valid region for sampling background patches is:
# - Within the breast tissue (breast_mask)
# - Outside the exclusion zone (NOT exclusion_mask)
# - Eroded to account for patch size (so patches don't extend beyond breast)

# %%
# Valid region = breast AND NOT exclusion
valid_mask = cv2.bitwise_and(breast_mask, cv2.bitwise_not(exclusion_mask))

print(f"Breast mask pixels: {np.sum(breast_mask > 0)}")
print(f"After excluding ROI zone: {np.sum(valid_mask > 0)}")

# %%
# Erode to ensure patch centres produce patches fully within breast
kernel_erode = np.ones((PATCH_SIZE // 2, PATCH_SIZE // 2), np.uint8)
valid_mask_eroded = cv2.erode(valid_mask, kernel_erode, iterations=1)

print(f"After erosion: {np.sum(valid_mask_eroded > 0)}")
print(f"Erosion kernel: {kernel_erode.shape} (half patch size)")

# %%
# Visualise the valid sampling region
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(valid_mask, cmap='gray')
axes[0].set_title("Valid Region (breast - exclusion)")
axes[0].axis('off')

axes[1].imshow(valid_mask_eroded, cmap='gray')
axes[1].set_title("Valid Region (eroded)")
axes[1].axis('off')

# Final overlay
axes[2].imshow(full_image, cmap='gray')
axes[2].imshow(valid_mask_eroded, cmap='Greens', alpha=0.4)
axes[2].imshow(exclusion_mask, cmap='Reds', alpha=0.3)
axes[2].set_title("Valid Sampling Region (green) vs Exclusion (red)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step 4: Random Sampling with Minimum Distance
#
# We randomly sample patch centres from the valid region, ensuring patches don't overlap (minimum distance = half patch size):

# %%
# Get all valid pixel coordinates
valid_points = np.where(valid_mask_eroded > 0)
print(f"Number of valid sampling points: {len(valid_points[0])}")

# These are (row, col) = (y, x) coordinates
print(f"Y range: [{valid_points[0].min()}, {valid_points[0].max()}]")
print(f"X range: [{valid_points[1].min()}, {valid_points[1].max()}]")

# %%
# Demonstrate the sampling process
np.random.seed(42)
sampled_positions = set()
num_bg_patches = 10
min_distance = PATCH_SIZE // 2

bg_centres = []
attempts = 0

while len(bg_centres) < num_bg_patches and attempts < num_bg_patches * 10:
    attempts += 1
    
    # Random sample from valid points
    idx = np.random.randint(0, len(valid_points[0]))
    cy, cx = valid_points[0][idx], valid_points[1][idx]
    
    # Check distance from already sampled positions
    too_close = any(
        abs(cx - sx) < min_distance and abs(cy - sy) < min_distance
        for (sx, sy) in sampled_positions
    )
    
    if too_close:
        continue
    
    bg_centres.append((cx, cy))
    sampled_positions.add((cx, cy))
    print(f"Patch {len(bg_centres)}: centre=({cx}, {cy}) [attempt {attempts}]")

print(f"\nTotal attempts: {attempts}")

# %%
# Visualise sampled centres on the mammogram
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(full_image, cmap='gray')
ax.imshow(valid_mask_eroded, cmap='Greens', alpha=0.2)
ax.imshow(exclusion_mask, cmap='Reds', alpha=0.2)

# Draw sampled patch centres and boundaries
for i, (cx, cy) in enumerate(bg_centres):
    rect = mpatches.Rectangle(
        (cx - half_patch, cy - half_patch), 
        PATCH_SIZE, PATCH_SIZE,
        linewidth=2, edgecolor='lime', facecolor='none'
    )
    ax.add_patch(rect)
    ax.plot(cx, cy, 'go', markersize=5)
    ax.annotate(str(i+1), (cx, cy), color='white', fontsize=8, ha='center', va='center')

ax.set_title("Sampled Background Patch Locations")
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Step 5: Extract Background Patch
#
# Extract a single background patch using the same boundary handling as ROI patches:

# %%
# Extract a single background patch (first sampled centre)
cx, cy = bg_centres[0]

# Calculate boundaries
x1, y1 = cx - half_patch, cy - half_patch
x2, y2 = x1 + PATCH_SIZE, y1 + PATCH_SIZE

# Clamp to image boundaries
x1_valid = max(0, x1)
y1_valid = max(0, y1)
x2_valid = min(img_w, x2)
y2_valid = min(img_h, y2)

# Extract patch
bg_patch = full_image[y1_valid:y2_valid, x1_valid:x2_valid].copy()

# Calculate padding if needed
pad_left = max(0, -x1)
pad_top = max(0, -y1)
pad_right = max(0, x2 - img_w)
pad_bottom = max(0, y2 - img_h)

if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
    bg_patch = cv2.copyMakeBorder(bg_patch, pad_top, pad_bottom, pad_left, pad_right,
                                   cv2.BORDER_CONSTANT, value=0)

print(f"Background patch shape: {bg_patch.shape}")
print(f"Padding: left={pad_left}, top={pad_top}, right={pad_right}, bottom={pad_bottom}")

# %%
# Visualise the extracted background patch
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(bg_patch, cmap='gray')
axes[0].set_title(f"Extracted Background Patch ({PATCH_SIZE}×{PATCH_SIZE})")
axes[0].axis('off')

# Verify no ROI overlap
mask_region = roi_mask[y1_valid:y2_valid, x1_valid:x2_valid]
roi_overlap = np.sum(mask_region > 0)

axes[1].imshow(bg_patch, cmap='gray')
if roi_overlap > 0:
    axes[1].imshow(mask_region, cmap='Reds', alpha=0.5)
axes[1].set_title(f"ROI pixels in patch: {roi_overlap} (should be 0)")
axes[1].axis('off')

plt.tight_layout()
plt.show()


# %% [markdown]
# Then to put all that into functions:

# %%
def get_breast_mask(img):
    """Get binary mask of breast region using thresholding."""
    if img.dtype != np.uint8:
        img_normalized = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img_normalized = img

    blur = cv2.GaussianBlur(img_normalized, (5, 5), 0)
    _, breast_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = np.ones((50, 50), np.uint8)
    opened_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    return closed_mask


def extract_background_patches(full_image, roi_masks, patch_size=224, num_patches=10):
    """
    Extract background patches from breast tissue avoiding ROI regions.
    """
    patches_list = []
    img_h, img_w = full_image.shape[:2]
    half_patch = patch_size // 2

    # Get breast tissue mask
    breast_mask = get_breast_mask(full_image)

    # Combine ROI masks into exclusion zone
    if roi_masks:
        if isinstance(roi_masks, list):
            exclusion_mask = np.zeros_like(roi_masks[0])
            for mask in roi_masks:
                exclusion_mask = cv2.bitwise_or(exclusion_mask, mask)
        else:
            exclusion_mask = roi_masks

        # Dilate exclusion zone
        kernel = np.ones((patch_size, patch_size), np.uint8)
        exclusion_mask = cv2.dilate(exclusion_mask, kernel, iterations=1)
    else:
        exclusion_mask = np.zeros((img_h, img_w), dtype=np.uint8)

    # Valid region = breast AND NOT exclusion
    valid_mask = cv2.bitwise_and(breast_mask, cv2.bitwise_not(exclusion_mask))

    # Erode to account for patch size
    kernel_erode = np.ones((patch_size // 2, patch_size // 2), np.uint8)
    valid_mask = cv2.erode(valid_mask, kernel_erode, iterations=1)

    valid_points = np.where(valid_mask > 0)
    if len(valid_points[0]) == 0:
        return []

    np.random.seed(42)
    sampled_positions = set()

    for _ in range(num_patches * 10):
        if len(patches_list) >= num_patches:
            break

        idx = np.random.randint(0, len(valid_points[0]))
        cy, cx = valid_points[0][idx], valid_points[1][idx]

        # Check distance from already sampled positions
        too_close = any(abs(cx - sx) < patch_size // 2 and abs(cy - sy) < patch_size // 2
                       for (sx, sy) in sampled_positions)
        if too_close:
            continue

        x1, y1 = cx - half_patch, cy - half_patch
        x2, y2 = x1 + patch_size, y1 + patch_size

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
            patch = cv2.copyMakeBorder(patch, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_CONSTANT, value=0)

        if patch.shape[0] != patch_size or patch.shape[1] != patch_size:
            patch = cv2.resize(patch, (patch_size, patch_size))

        patches_list.append((patch, (cx, cy)))
        sampled_positions.add((cx, cy))

    return patches_list


# %%
# Extract background patches
bg_patches = extract_background_patches(full_image, [roi_mask], patch_size=PATCH_SIZE, num_patches=10)
print(f"Extracted {len(bg_patches)} background patches")

# %%
# Visualize background patches
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

for i, (patch, (px, py)) in enumerate(bg_patches):
    axes[i].imshow(patch, cmap='gray')
    axes[i].set_title(f"Background {i+1}", fontsize=9)
    axes[i].axis('off')

plt.suptitle(f"Background Patches (Class 0: {CLASS_NAMES[0]})", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Visualize All Patch Locations

# %%
fig, ax = plt.subplots(figsize=(8, 10))
ax.imshow(full_image, cmap='gray')
ax.imshow(roi_mask, cmap='Reds', alpha=0.3)

# Draw ROI patches in red
for i, (patch, (px, py)) in enumerate(roi_patches):
    half = PATCH_SIZE // 2
    rect = mpatches.Rectangle((px - half, py - half), PATCH_SIZE, PATCH_SIZE,
                                linewidth=2, edgecolor='red', facecolor='none')
    ax.add_patch(rect)

# Draw background patches in green
for i, (patch, (px, py)) in enumerate(bg_patches):
    half = PATCH_SIZE // 2
    rect = mpatches.Rectangle((px - half, py - half), PATCH_SIZE, PATCH_SIZE,
                                linewidth=2, edgecolor='green', facecolor='none')
    ax.add_patch(rect)

# Legend
roi_patch = mpatches.Patch(edgecolor='red', facecolor='none', linewidth=2, label='ROI Patches')
bg_patch = mpatches.Patch(edgecolor='green', facecolor='none', linewidth=2, label='Background Patches')
ax.legend(handles=[roi_patch, bg_patch], loc='upper right')

ax.set_title("ROI and Background Patch Locations")
ax.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Augmentation Pipeline for Patch Training
#
# The augmentation pipeline for patches is lighter than for full images since patches are already 224×224.

# %%
# Convert patch to RGB for augmentation
sample_patch = roi_patches[0][0]
patch_rgb = np.stack([sample_patch] * 3, axis=-1)

# %%
train_transform = get_train_transform(output_size=224)

# Show multiple augmented versions
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
axes = axes.flatten()

axes[0].imshow(patch_rgb)
axes[0].set_title("Original", fontsize=10)
axes[0].axis('off')

for i in range(1, 10):
    augmented = train_transform(image=patch_rgb)['image']
    axes[i].imshow(augmented)
    axes[i].set_title(f"Aug {i}", fontsize=10)
    axes[i].axis('off')

plt.suptitle("Patch Augmentation Examples", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Validation Transform
#
# For validation/inference, only resize is applied (no augmentation).

# %%
val_transform = get_val_transform(output_size=224)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(patch_rgb)
axes[0].set_title("Original Patch (224×224)")
axes[0].axis('off')

val_patch = val_transform(image=patch_rgb)['image']
axes[1].imshow(val_patch)
axes[1].set_title("Validation Transform (224×224)")
axes[1].axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Compare ROI vs Background Patches
#
# Visual comparison of ROI patches (containing lesions) vs background patches (normal tissue).

# %%
fig, axes = plt.subplots(2, 5, figsize=(15, 6))

# Top row: ROI patches
for i in range(min(5, len(roi_patches))):
    patch, _ = roi_patches[i]
    axes[0, i].imshow(patch, cmap='gray')
    axes[0, i].set_title(f"ROI {i+1}", fontsize=10)
    axes[0, i].axis('off')
axes[0, 0].set_ylabel("ROI\n(Malignant Mass)", fontsize=10, rotation=0, ha='right', va='center')

# Bottom row: Background patches
for i in range(min(5, len(bg_patches))):
    patch, _ = bg_patches[i]
    axes[1, i].imshow(patch, cmap='gray')
    axes[1, i].set_title(f"BG {i+1}", fontsize=10)
    axes[1, i].axis('off')
axes[1, 0].set_ylabel("Background\n(Normal)", fontsize=10, rotation=0, ha='right', va='center')

plt.suptitle("ROI Patches vs Background Patches", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
