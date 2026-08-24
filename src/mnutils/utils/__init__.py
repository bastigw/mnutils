"""Loading, file discovery, NIfTI and spectral helpers."""

from . import data_loaders, file_helpers, images, nifti, partial_volume, spectra
from .data_loaders import load_mat_file
from .file_helpers import (
    get_exam_folder,
    get_h5_data_from_series,
    get_mat_data_from_series,
)
from .partial_volume import mask_occupancy
from .spectra import calculate_ppm_axis

__all__ = [
    # modules
    "data_loaders",
    "file_helpers",
    "images",
    "nifti",
    "partial_volume",
    "spectra",
    # frequently used helpers
    "calculate_ppm_axis",
    "get_exam_folder",
    "get_h5_data_from_series",
    "get_mat_data_from_series",
    "load_mat_file",
    "mask_occupancy",
]
