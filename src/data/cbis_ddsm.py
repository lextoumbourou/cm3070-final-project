"""
CBIS-DDSM dataset helper library.

Provides common utilities for working with CBIS-DDSM DICOM data.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
from pydantic import BaseModel


class DCMData(BaseModel):
    """Parsed DICOM file path data from CBIS-DDSM CSV columns."""

    subject_id: str
    study_uid: str
    series_uid: str
    dcm_file: str


def parse_dcm_path(dcm_path: str) -> DCMData:
    """
    Parse DICOM file path string to extract components.

    The CBIS-DDSM CSV files contain paths in the format:
    subject_id/study_uid/series_uid/filename.dcm
    """
    parts = str(dcm_path).strip().split("/")
    dcm_file = parts[-1].strip().split(".")[0]
    return DCMData(
        subject_id=parts[0], study_uid=parts[1], series_uid=parts[2], dcm_file=dcm_file
    )


def resolve_dcm_path(dcm_data: DCMData, metadata_df: pd.DataFrame, dataset_root: Path) -> Path:
    """
    Resolve the full filesystem path for a DICOM file.

    Uses the metadata CSV to find the actual file location, since the paths
    in the case description CSVs don't match the filesystem layout directly.
    """
    meta = metadata_df[
        (metadata_df["Subject ID"] == dcm_data.subject_id)
        & (metadata_df["Series UID"] == dcm_data.series_uid)
        & (metadata_df["Study UID"] == dcm_data.study_uid)
    ].iloc[0]
    file_location = meta["File Location"]
    return dataset_root / Path(file_location) / (dcm_data.dcm_file + ".dcm")


def load_dicom_array(file_path: Path) -> np.ndarray:
    """
    Load a DICOM file and return pixel data as numpy array.
    """
    ds = pydicom.dcmread(file_path)
    return ds.pixel_array
