"""Loading, file discovery, NIfTI and spectral helpers."""

from . import data_loaders, file_helpers, images, nifti, spectra
from .data_loaders import load_mat_file
from .file_helpers import (
    get_exam_folder,
    get_h5_data_from_series,
    get_mat_data_from_series,
)
from .spectra import calculate_ppm_axis

__all__ = [
    # modules
    "data_loaders",
    "file_helpers",
    "images",
    "nifti",
    "spectra",
    # frequently used helpers
    "calculate_ppm_axis",
    "get_exam_folder",
    "get_h5_data_from_series",
    "get_mat_data_from_series",
    "load_mat_file",
]
