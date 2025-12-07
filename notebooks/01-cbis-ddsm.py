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
# ## Imports

# %%
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# %% [markdown]
# ## Metadata File Review
#
# There is 2 files provided for each split:
#
# ### Train
#
# - `datasets/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv`
# - `datasets/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv`
#
# ### Test
#
# - `datasets/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv`
# - `datasets/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_test_set.csv`
#
#
# And 2 meta files:
#
# - `datasets/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv`
# - `datasets/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv`

# %%
train_set_mass_df = pd.read_csv("../datasets/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_train_set.csv")
train_set_mass_df.head()

# %%
test_set_mass_df = pd.read_csv("../datasets/cbis-ddsm-breast-cancer-image-dataset/csv/mass_case_description_test_set.csv")

# %%
train_set_calc_df = pd.read_csv("../datasets/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_train_set.csv")
train_set_calc_df.head()

# %%
test_set_calc_df = pd.read_csv("../datasets/cbis-ddsm-breast-cancer-image-dataset/csv/calc_case_description_test_set.csv")

# %%
dicom_info_df = pd.read_csv("../datasets/cbis-ddsm-breast-cancer-image-dataset/csv/dicom_info.csv")
dicom_info_df.head()

# %%
meta_df = pd.read_csv("../datasets/cbis-ddsm-breast-cancer-image-dataset/csv/meta.csv")
meta_df.head()

# %% [markdown]
# The paper claims that there's 891 mass cases, for some reason we appear to have 892 in the dataset.

# %%
len(set(train_set_mass_df.patient_id.unique()) | set(test_set_mass_df.patient_id.unique()))

# %% [markdown]
# However, we have the correct number of calcification cases.

# %%
test_set_calc_df.patient_id.nunique() + train_set_calc_df.patient_id.nunique()

# %% [markdown]
# Let's have a closer look at one patient id to start with. Later we'll try to understand stastics for all patient ids.

# %%
patient_5 = train_set_calc_df[train_set_calc_df.patient_id == "P_00005"]

# %%
patient_5

# %% [markdown]
# For this patient, we have a mammography for a single breast, with the expected two views:
# - CC - a top-to-bottom view.
# - MLO - a side view.

# %% [markdown]
# ## Image Extraction
#
# Each row, which represents a view of the patient's breast contains 3 dicom files:
#
# - image file path
# - ROI mask file path
# - cropped image file path
#
# The creator of the CBIS-DDSM JPG dataset has converted each of these files into jpg files, which can be retrieve by extracting the dicom id from the path, and then looking it up in the `dicom_info_df` file.
#
# Firstly, need a function to extract the Dicom id from the path

# %%

# %% [markdown]
# ### Fetch img file path

# %% [markdown]
# From some manual exploration, I put together a function that extracts the image file given a row from the dataset.

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
# At this point, I'd like to know, do all patients have both images, or are there some patients with an incomplete mammogram?

# %% [markdown]
# ### Fetch ROI mask file path

# %% [markdown]
#

# %%
mask_img_path = get_img_path(patient_5.iloc[0]["ROI mask file path"])
mask_img = Image.open(mask_img_path)

# %%
plt.imshow(mask_img, cmap="grey")

# %%
patient_np = np.array(patient_img)
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
# ## EDA
#
# Firstly, we'll take a look at the label distribution.

# %% [markdown]
# ### Combine all data for label analysis

# %%
# Combine train and test sets for comprehensive analysis
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
# ### Image View Distribution

# %%
fig, ax = plt.subplots(figsize=(10, 6))

view_counts = all_data_df['image view'].value_counts()
ax.bar(view_counts.index, view_counts.values, color='#d35400')
ax.set_title('Image View Distribution', fontsize=12, fontweight='bold')
ax.set_xlabel('Image View')
ax.set_ylabel('Count')
for i, v in enumerate(view_counts.values):
    ax.text(i, v + 5, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# %%
