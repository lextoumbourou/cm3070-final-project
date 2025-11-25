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
# # CBIS-DDSM Original
#
# This is the complete dataset, downloaded from https://www.cancerimagingarchive.net/collection/cbis-ddsm/

# %%
from pathlib import Path

import pandas as pd
import pydicom
import matplotlib.pyplot as plt

# %%
DATASET_ROOT = Path("../datasets/CBIS-DDSM")
IMG_ROOT = DATASET_ROOT / "CBIS-DDSM"

# %%
train_mass_df = pd.read_csv(DATASET_ROOT / "mass_case_description_train_set.csv")
train_mass_df.head()

# %%
train_calc_df = pd.read_csv(DATASET_ROOT / "calc_case_description_train_set.csv")
train_calc_df.head()

# %%
raw_img = train_calc_df.iloc[0]["image file path"].strip()

# %%
raw_img

# %%
roi_mask_path_img = train_calc_df.iloc[0]["ROI mask file path"].strip()
roi_mask_path_img

# %%
cropped_img = train_calc_df.iloc[0]["cropped image file path"].strip()
cropped_img

# %%
cropped_subject_id = cropped_img.split("/")[0]
cropped_subject_id

# %%
cropped_study_uid = cropped_img.split("/")[1]
cropped_study_uid

# %%
cropped_series_uid = cropped_img.split("/")[2]
cropped_series_uid

# %%
img_num = int(cropped_img.split("/")[-1].replace(".dcm", "")[-1]) + 1
img_num

# %%
metadata_df = pd.read_csv(DATASET_ROOT / "metadata.csv")

# %%
metadata_df["Subject ID"].nunique()

# %%
len(metadata_df)

# %%
cropped_meta = metadata_df[
    (metadata_df["Subject ID"] == cropped_subject_id) &
    (metadata_df["Series UID"] == cropped_series_uid) &
    (metadata_df["Study UID"] == cropped_study_uid)
].iloc[0]

# %%
dict(cropped_meta)

# %%
img_path = DATASET_ROOT / cropped_meta["File Location"]
img_path

# %%
files = list(img_path.rglob("[!.]*.dcm"))
list(files)

# %%
ds = pydicom.dcmread(files[0])
img = ds.pixel_array

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# %%
ds = pydicom.dcmread(files[1])
img = ds.pixel_array

plt.figure(figsize=(6, 6))
plt.imshow(img, cmap="gray")
plt.axis("off")
plt.show()

# %%
