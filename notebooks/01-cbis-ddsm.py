# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
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
# # CBIS-DDSM EDA

# %% [markdown]
# This notebook describes my exploratory dataset analysis of the CBIS-DDSM dataset (Lee et al).
#
# Firstly, we import some libraries used throughout the notebook.

# %% [markdown]
# ## Imports

# %%
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2
import matplotlib.patches as patches

# %%
DATASET_ROOT = Path("../datasets/CBIS-DDSM")
IMG_ROOT = DATASET_ROOT / "CBIS-DDSM"

# %% [markdown]
# ## Dataset Overview

# %% [markdown]
# The dataset was downloaded from from https://www.cancerimagingarchive.net/collection/cbis-ddsm.
#
# It is a 164GB compressed dataset which uncompresses to around 180GB.

# %%
# !cd {DATASET_ROOT} && du -sh *

# %% [markdown]
# ## Metadata File Review
#
# There is 2 files provided for each split, representing either calcification or mass abnormalities found in the breast.
#
# - `calc_case_description_${train|test}_set.csv`
# - `mass_case_description_${train|test}_set.csv`
#
# Here we load each file, then concat together to give us one dataset file per split.

# %%
train_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_train_set.csv")
train_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_train_set.csv")
train_df = pd.concat([train_mass_df, train_calc_df])
train_mass_df = train_calc_df = None
train_df.head(1)

# %%
test_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_test_set.csv")
test_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_test_set.csv")
test_df = pd.concat([test_mass_df, test_calc_df])
test_mass_df = test_mass_df = None
test_df.head(1)

# %%
all_df = pd.concat([train_df, test_df])

# %%
metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")
metadata_df.head(1)

# %% [markdown]
# The paper claims that there's 891 mass cases, although the actual dataset appears to have 892 mass abnormalities.

# %%
len(all_df[all_df["abnormality type"] == "mass"].patient_id.unique())

# %% [markdown]
# The paper also describes 753 calcification abnormalities, which matches what we see in the paper.

# %%
len(all_df[all_df["abnormality type"] == "calcification"].patient_id.unique())

# %% [markdown]
# Let's have a closer look at one patient id to start with. Later we'll try to understand stastics for all patient ids.

# %%
patient_5 = train_set_calc_df[train_set_calc_df.patient_id == "P_00005"]

# %%
patient_5

# %% [markdown]
# For this patient, we have a mammography for a single breast, with the expected two views:
#
# - CC (Caniocaudal): a top-to-bottom view.
# - MLO (Mediolateral): a side view.

# %% [markdown]
# ## Image Extraction
#
# Each row, which represents a view of the patient's breast contains references to 3 dicom files:
#
# - image file path
# - ROI mask file path
# - cropped image file path
#
# The dataset I'm using is actually a post-processed version of CBIS-DDSM, [found on Kaggle by @awsaf](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset), which has converted each of the original Dicom files in JPG. The image path which can be retrieve by extracting the dicom id from the path, and then looking it up in the `dicom_info_df` file.

# %% [markdown]
# ### Fetch img file path

# %% [markdown]
# From some manual exploration, I put together some helper functions that extracts the image file given a row from the dataset.

# %%
JPEG_ROOT = Path("../datasets/cbis-ddsm-breast-cancer-image-dataset/jpeg")

def get_img_id_from_dcm_file(path: Path):
    return str(path).split("/")[1]

def get_jpg_path(img_file_path: str):
    return JPEG_ROOT / img_file_path.replace("CBIS-DDSM/jpeg/", "")

def get_img_path(img_path):
    img_file = get_img_id_from_dcm_file(img_path)
    dicom_row = dicom_info_df[dicom_info_df.StudyInstanceUID == img_file].iloc[0]
    return get_jpg_path(dicom_row.image_path)

def get_patient_img(image_file_path):
    return Image.open(get_img_path(image_file_path))


def show_img_grid(cc_img, mlo_img):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    
    axes[0].imshow(cc_img, cmap='gray')
    axes[0].set_title('CC View')
    axes[0].axis('off')
    
    axes[1].imshow(mlo_img, cmap='gray')
    axes[1].set_title('MLO View')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


# %%
cc_img = get_patient_img(patient_5.iloc[0]["image file path"])
mlo_img = get_patient_img(patient_5.iloc[1]["image file path"])

# %%
show_img_grid(cc_img, mlo_img)

# %% [markdown]
# And another example:

# %%
patient_02033 = train_set_mass_df[train_set_mass_df.patient_id == "P_02033"].reset_index(drop=True)
patient_02033

# %%
cc_img = get_patient_img(patient_02033[patient_02033["image view"] == "CC"].iloc[0]["image file path"])
mlo_img = get_patient_img(patient_02033[patient_02033["image view"] == "MLO"].iloc[0]["image file path"])
show_img_grid(cc_img, mlo_img)

# %% [markdown]
# ### Fetch ROI mask file path

# %% [markdown]
# Each mammograph view also contains a region-of-interest annotation, which can extract either a calcification or mass abnormality within the breast. We can visual them as follows.

# %%
mask_img_path = get_img_path(patient_5[patient_5["image view"] == "CC"].iloc[0]["ROI mask file path"])
mask_img = Image.open(mask_img_path)

# %%
plt.imshow(mask_img, cmap="grey")

# %%
patient_np = np.array(cc_img)
mask_np = np.array(mask_img)

plt.figure(figsize=(8,8))
plt.imshow(patient_np, cmap='gray')

# overlay mask with transparency
plt.imshow(mask_np, cmap='jet', alpha=0.4)

plt.axis("off")
plt.title("ROI Overlay")
plt.show()


# %% [markdown]
# ### Fetch cropped img path
#
# To get the cropped image, it seems we need to fetch it relative to the ROI mask.

# %%
def get_cropped_img_from_roi_path(mask_img_path):
    parent_dir = mask_img_path.parent 
    files = list(parent_dir.glob("*.jpg"))
    crop = [f for f in files if f.name.startswith("2-")][0]
    return crop


# %%
cropped_img_path = get_cropped_img_from_roi_path(mask_img_path)
cropped_img_path

# %%
plt.imshow(Image.open(cropped_img_path), cmap="grey")


# %% [markdown]
# ## Manual Breast Extraction
#
# Since we want to explore the possibility of utilising this model again Mammogram data that does not contain region-of-interest annotations, I want to explore an alternate method of breast extraction, which aims to remove blank space, and non-breast noise from the image using thresholding. Some of this work builds upon the knowledge that I gain from studying signal processing in CM3065 - Intelligent Signal Processing.
#
# To attempt to crop some of the area around the breast, I'll utilise some classical object detection approaches.
#
# I created some utility functions.

# %%
def show_img_grid(img_1, img_2):
    fig, axes = plt.subplots(1, 2, figsize=(6, 4))

    axes[0].imshow(img_1, cmap='gray')
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    axes[1].imshow(img_2, cmap='gray')
    axes[1].set_xticks([])
    axes[1].set_yticks([])

    plt.tight_layout()
    plt.show()


# %%
sample_img = np.array(cc_img.convert('L'))

# %% [markdown]
# ### Gaussian Blur
#
# A 5×5 Gaussian blur smooths out high-frequency noise while preserving the overall structure. This prevents small noise pixels from creating spurious regions during thresholding.
#
# Source: https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html

# %%
blurred_img = cv2.GaussianBlur(sample_img, (5, 5), 0)
show_img_grid(sample_img, blurred_img)

# %% [markdown]
# ### Otsu's Thresholding
#
# Otsu's method automatically determines the optimal threshold value by minimising intra-class variance (or equivalently, maximising inter-class variance) between foreground and background pixels. This is ideal for separating the breast tissue from the dark background without manual threshold tuning.
#
# Source: https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

# %%
_, breast_mask = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
show_img_grid(blurred_img, breast_mask)


# %% [markdown]
# ### Morphological Transformations
#
# Morphological operations refine the binary mask by removing noise and filling gaps:
#
# - **Opening** (erosion followed by dilation): Removes small bright spots/noise outside the breast region
# - **Closing** (dilation followed by erosion): Fills small holes within the breast region
#
# The 100×100 kernel size is chosen to handle the large scale of mammogram images (typically 3000-5000 pixels). Multiple iterations of closing ensure continuous breast boundaries.

# %%
def apply_morphological_transforms(
    thresh_frame, iterations: int = 2
):
    kernel = np.ones((100, 100), np.uint8)
    opened_mask = cv2.morphologyEx(thresh_frame, cv2.MORPH_OPEN, kernel)
    closed_mask = cv2.morphologyEx(opened_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return closed_mask


# %%
morph_img = apply_morphological_transforms(breast_mask)
show_img_grid(breast_mask, morph_img)


# %% [markdown]
# As we can see, it has removed the noisy label.
#
# Next we can find countours, and use the max contour as the overall bounding box.

# %%
def get_contours_from_mask(mask):
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x, y, w, h)


# %%
x, y, w, h = get_contours_from_mask(morph_img)

plt.imshow(sample_img, cmap='gray')
ax = plt.gca()

rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor='r', facecolor='none')
ax.add_patch(rect)

plt.title('Original with ROI')
plt.axis('off')
plt.show()

# %% [markdown]
# Very nice, near perfect bounding box around the breast area.

# %% [markdown]
# ## Metadata EDA
#
# Firstly, we'll take a look some of the provided metadata, to better understand the CBIS-DDSM dataset.

# %% [markdown]
# ### Combine all data for label analysis

# %% [markdown]
# Firstly we combine the mass and calcification datasets into a single dataset for analysis. We add a label to tell us which dataset it came from.

# %%
all_mass_df = pd.concat([train_set_mass_df, test_set_mass_df], ignore_index=True)
all_calc_df = pd.concat([train_set_calc_df, test_set_calc_df], ignore_index=True)

# Add abnormality type column for combined analysis
all_mass_df['abnormality_category'] = 'mass'
all_calc_df['abnormality_category'] = 'calcification'

# Combine both mass and calc data
all_data_df = pd.concat([all_mass_df, all_calc_df], ignore_index=True)

print(f"Total number of cases: {len(all_data_df)}")
print(f"Mass cases: {len(all_mass_df)}")
print(f"Calcification cases: {len(all_calc_df)}")

# %% [markdown]
# ### Pathology Distribution
#
# The most important label is `pathology`, which indicates whether the abnormality is benign or malignant.
#
# We explore the distribution over each dataset.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Overall pathology distribution
pathology_counts = all_data_df['pathology'].value_counts()
axes[0].bar(pathology_counts.index, pathology_counts.values, color=['#2ecc71', '#3498db', '#e74c3c'])
axes[0].set_title('Overall Pathology Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Pathology')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=45)
for i, v in enumerate(pathology_counts.values):
    axes[0].text(i, v + 5, str(v), ha='center', va='bottom')

# Mass pathology distribution
mass_pathology = all_mass_df['pathology'].value_counts()
axes[1].bar(mass_pathology.index, mass_pathology.values, color=['#2ecc71', '#3498db', '#e74c3c'])
axes[1].set_title('Mass Pathology Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Pathology')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(mass_pathology.values):
    axes[1].text(i, v + 5, str(v), ha='center', va='bottom')

# Calcification pathology distribution
calc_pathology = all_calc_df['pathology'].value_counts()
axes[2].bar(calc_pathology.index, calc_pathology.values, color=['#2ecc71', '#3498db', '#e74c3c'])
axes[2].set_title('Calcification Pathology Distribution', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Pathology')
axes[2].set_ylabel('Count')
axes[2].tick_params(axis='x', rotation=45)
for i, v in enumerate(calc_pathology.values):
    axes[2].text(i, v + 5, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### BI-RADS Assessment Distribution
#
# The BI-RADS assessment category (0-5) indicates the level of suspicion.

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall assessment distribution
assessment_counts = all_data_df['assessment'].value_counts().sort_index()
axes[0].bar(assessment_counts.index.astype(str), assessment_counts.values, color='#9b59b6')
axes[0].set_title('BI-RADS Assessment Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Assessment Category')
axes[0].set_ylabel('Count')
for i, (idx, v) in enumerate(assessment_counts.items()):
    axes[0].text(i, v + 5, str(v), ha='center', va='bottom')

# Assessment by pathology
pathology_assessment = pd.crosstab(all_data_df['assessment'], all_data_df['pathology'])
pathology_assessment.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#3498db', '#e74c3c'])
axes[1].set_title('BI-RADS Assessment by Pathology', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Assessment Category')
axes[1].set_ylabel('Count')
axes[1].legend(title='Pathology', bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].tick_params(axis='x', rotation=0)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Abnormality Type Distribution

# %% [markdown]
# Next we breakdown the abnormality type.
#
# Pie chart source: https://matplotlib.org/stable/gallery/pie_and_polar_charts/pie_features.html#sphx-glr-gallery-pie-and-polar-charts-pie-features-py

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Abnormality category
abnormality_counts = all_data_df['abnormality_category'].value_counts()
axes[0].pie(abnormality_counts.values, labels=abnormality_counts.index, autopct='%1.1f%%',
            colors=['#e67e22', '#16a085'], startangle=90)
axes[0].set_title('Mass vs Calcification Distribution', fontsize=12, fontweight='bold')

# Abnormality type (from the original column)
abnormality_type_counts = all_data_df['abnormality type'].value_counts()
axes[1].bar(abnormality_type_counts.index, abnormality_type_counts.values, color='#34495e')
axes[1].set_title('Abnormality Type Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Abnormality Type')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
for i, v in enumerate(abnormality_type_counts.values):
    axes[1].text(i, v + 5, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Mass Shape and Margins Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Mass shape
mass_shape_counts = all_mass_df['mass shape'].value_counts()
axes[0].barh(mass_shape_counts.index, mass_shape_counts.values, color='#e67e22')
axes[0].set_title('Mass Shape Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Mass Shape')
for i, v in enumerate(mass_shape_counts.values):
    axes[0].text(v + 2, i, str(v), va='center')

# Mass margins
mass_margins_counts = all_mass_df['mass margins'].value_counts()
axes[1].barh(mass_margins_counts.index, mass_margins_counts.values, color='#16a085')
axes[1].set_title('Mass Margins Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Mass Margins')
for i, v in enumerate(mass_margins_counts.values):
    axes[1].text(v + 2, i, str(v), va='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Calcification Type and Distribution

# %%
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Calc type
calc_type_counts = all_calc_df['calc type'].value_counts()
axes[0].barh(calc_type_counts.index, calc_type_counts.values, color='#8e44ad')
axes[0].set_title('Calcification Type Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Count')
axes[0].set_ylabel('Calcification Type')
for i, v in enumerate(calc_type_counts.values):
    axes[0].text(v + 2, i, str(v), va='center')

# Calc distribution
calc_dist_counts = all_calc_df['calc distribution'].value_counts()
axes[1].barh(calc_dist_counts.index, calc_dist_counts.values, color='#c0392b')
axes[1].set_title('Calcification Distribution Pattern', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Count')
axes[1].set_ylabel('Distribution Pattern')
for i, v in enumerate(calc_dist_counts.values):
    axes[1].text(v + 2, i, str(v), va='center')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Breast Density Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 6))

# Handle both column name formats
density_col = 'breast_density' if 'breast_density' in all_data_df.columns else 'breast density'
breast_density_counts = all_data_df[density_col].value_counts().sort_index()

ax.bar(breast_density_counts.index.astype(str), breast_density_counts.values, color='#2980b9')
ax.set_title('Breast Density Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Breast Density')
ax.set_ylabel('Count')
for i, v in enumerate(breast_density_counts.values):
    ax.text(i, v + 5, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Subtlety Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 6))

subtlety_counts = all_data_df['subtlety'].value_counts().sort_index()
ax.bar(subtlety_counts.index.astype(str), subtlety_counts.values, color='#27ae60')
ax.set_title('Subtlety Rating Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Subtlety (1=Subtle, 5=Obvious)')
ax.set_ylabel('Count')
for i, v in enumerate(subtlety_counts.values):
    ax.text(i, v + 5, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Image Dimensions Analysis
#
# Finally, we analysis the height and width distribution of images across the dataset, to better understand the image sizes in the CBIS-DDSM dataste.

# %%
from tqdm import tqdm

def get_image_dimensions(df, desc="Processing"):
    """Extract image dimensions for all images in a dataframe."""
    dimensions = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc=desc):
        try:
            img_path = get_img_path(row["image file path"])
            img = Image.open(img_path)
            width, height = img.size
            dimensions.append({
                "patient_id": row["patient_id"],
                "image_view": row["image view"],
                "width": width,
                "height": height,
                "aspect_ratio": width / height
            })
        except Exception as e:
            print(f"Error processing {row['patient_id']}: {e}")
    return pd.DataFrame(dimensions)

# %%
dimensions_df = get_image_dimensions(all_data_df, desc="Analysing image dimensions")

# %%
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Width distribution
axes[0].hist(dimensions_df['width'], bins=50, color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_title('Image Width Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Width (pixels)')
axes[0].set_ylabel('Count')
axes[0].axvline(dimensions_df['width'].median(), color='red', linestyle='--',
                label=f"Median: {dimensions_df['width'].median():.0f}")
axes[0].legend()

# Height distribution
axes[1].hist(dimensions_df['height'], bins=50, color='#2ecc71', edgecolor='black', alpha=0.7)
axes[1].set_title('Image Height Distribution', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Height (pixels)')
axes[1].set_ylabel('Count')
axes[1].axvline(dimensions_df['height'].median(), color='red', linestyle='--',
                label=f"Median: {dimensions_df['height'].median():.0f}")
axes[1].legend()

# Aspect ratio distribution
axes[2].hist(dimensions_df['aspect_ratio'], bins=50, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[2].set_title('Aspect Ratio Distribution (Width/Height)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Aspect Ratio')
axes[2].set_ylabel('Count')
axes[2].axvline(dimensions_df['aspect_ratio'].median(), color='red', linestyle='--',
                label=f"Median: {dimensions_df['aspect_ratio'].median():.2f}")
axes[2].legend()

plt.tight_layout()
plt.show()


# %%
