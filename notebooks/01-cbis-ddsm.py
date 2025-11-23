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
patient_5.iloc[0]


# %% [markdown]
# ### Retrieving files
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
def get_img_id_from_dcm_file(path: Path):
    return str(path).split("/")[1]


# %% [markdown]
# ### Fetch img file path

# %% [markdown]
# Now to fetch the id for the "image file path":

# %%
patient_5_img_file = patient_5.iloc[0]["image file path"]
patient_5_img_file

# %%
patient_5_img_id = get_img_id_from_dcm_file(patient_5_img_file)
patient_5_img_id

# %%
dicom_5_row = dicom_info_df[dicom_info_df.StudyInstanceUID == patient_5_id].iloc[0]
dicom_5_row.image_path

# %%
JPEG_ROOT = Path("../datasets/cbis-ddsm-breast-cancer-image-dataset/jpeg")


# %% [markdown]
# Now a function to normalise the JPEG path to use my path:

# %%
def get_jpg_path(img_file_path: str):
    return JPEG_ROOT / img_file_path.replace("CBIS-DDSM/jpeg/", "")


# %%
patient_img = Image.open(get_jpg_path(dicom_5_row.image_path))

# %%
plt.imshow(patient_img, cmap="grey")


# %% [markdown]
# Now to wrap all that in a single function.

# %%
def get_img_path(img_path):
    img_file = get_img_id_from_dcm_file(img_path)
    dicom_row = dicom_info_df[dicom_info_df.StudyInstanceUID == img_file].iloc[0]
    return get_jpg_path(dicom_row.image_path)


# %% [markdown]
# ### Fetch ROI mask file path

# %%
mask_img_path = get_img_path(patient_5.iloc[0]["ROI mask file path"])
mask_img = Image.open(mask_img_path)

# %%
plt.imshow(mask_img, cmap="grey")

# %%
patient_np = np.array(patient_img)
mask_np = np.array(mask)

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
