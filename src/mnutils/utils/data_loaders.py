import getpass
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from . import file_helpers


def load_mat_file(file_path: Path | str) -> dict:
    """Load a .mat file and return its contents as a dictionary.

    Transparently falls back from `scipy.io.loadmat` (plain v5/v7 files) to
    `mat73` (v7.3, HDF5-based files) when the former can't read the file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .mat file.

    Returns
    -------
    dict
        Contents of the .mat file.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist or is not a file.
    ValueError
        If `file_path` does not have a `.mat` extension.
    RuntimeError
        If the file can't be loaded by either scipy or mat73.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"The specified path does not exist or is not a file: '{file_path}'"
        )
    if file_path.suffix.lower() != ".mat":
        raise ValueError(f"The file does not have a .mat extension: '{file_path}'")

    try:
        # Try loading with scipy.io.loadmat first
        return _loadmat(file_path)
    except (NotImplementedError, ValueError):
        # If it fails, try loading with mat73 (for v7.3 .mat files)
        try:
            import mat73

            return mat73.loadmat(file_path)
        except ImportError as e:
            raise RuntimeError(
                "mat73 is required to read MATLAB v7.3 files but could not be imported."
            ) from e
        except OSError as e:
            logger.error(f"Failed to load .mat file with both scipy and mat73: {e}")
            raise OSError(f"Failed to open .mat file with mat73: {e}")
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Unexpected error loading .mat file with mat73: {e}")
    except (OSError, TypeError) as e:
        logger.error(f"Failed to load .mat file with scipy: {e}.")
        raise RuntimeError(f"Unexpected error loading .mat file with scipy: {e}")


def load_raw_fids(
    DATA_FOLDER: Path | str,
    series_id: int,
    load_most_recent: bool = True,
    force_override: bool = False,
) -> tuple[np.ndarray, Path]:
    """Load raw FIDs for a series, reading from a cached HDF5 file when available.

    Parameters
    ----------
    DATA_FOLDER : Path or str
        Path to the exam's data folder.
    series_id : int
        Series number to load raw FIDs for.
    load_most_recent : bool
        If True, return the most recently created cached `*_raw_fids.h5` file
        instead of re-reading the scan archive.
    force_override : bool
        If True, delete and regenerate an existing output HDF5 file rather than
        returning its cached contents.

    Returns
    -------
    tuple[np.ndarray, Path]
        The complex raw FIDs array and the path to the HDF5 file it was
        loaded from (or written to).
    """
    scan_archive_file = file_helpers.get_h5_data_from_series(DATA_FOLDER, series_id)
    series_folder = file_helpers.get_exam_folder(DATA_FOLDER) / f"Series{series_id}"

    # Find if there is already an h5 file with raw fids
    if load_most_recent:
        existing_files = list(series_folder.glob(f"Series{series_id}_raw_fids.h5"))
        if existing_files:
            # Get the most recent file based on creation time
            output_h5_file = max(existing_files, key=lambda f: f.stat().st_ctime)
            logger.debug(
                f"Found existing raw FIDs file: {output_h5_file}. Loading data from it."
            )
            with h5py.File(output_h5_file, "r") as hf:
                data = np.array(hf["fids"])
            return data, output_h5_file
        else:
            logger.debug(
                f"No existing raw FIDs file found in {series_folder}. Will create a new one."
            )

    output_h5_file = series_folder / f"Series{series_id}_raw_fids.h5"

    logger.debug(f"Loading raw FIDs from {scan_archive_file} into {output_h5_file}")

    if output_h5_file.exists():
        if not force_override:
            logger.warning(
                f"Output file {output_h5_file} already exists. Returning data from it."
            )
            with h5py.File(output_h5_file, "r") as hf:
                data = np.array(hf["fids"])
            return data, output_h5_file
        else:
            logger.warning(
                f"Output file {output_h5_file} already exists. "
                "Overriding as force_override is True."
            )
            output_h5_file.unlink()

    from .. import matlab

    matengine = matlab.connect_to_matlab()
    matlab.setup_util_path(matengine)
    matlab.add_matlablatest_path(matengine)
    matengine.manage_paths("ORCHESTRASDK", True, nargout=0)
    data = matengine.read_archive(str(scan_archive_file), 0, nargout=1)
    matengine.quit()
    logger.debug(
        f"Raw data loaded from MATLAB engine, type: {type(data)}. Matlabengine session closed."
    )
    data = np.asarray(data)
    # Validate data shape and type to be complex numpy array
    if not isinstance(data, np.ndarray) or not np.iscomplexobj(data):
        raise ValueError(
            f"Loaded data is not a complex numpy array. Got type: {type(data)}"
        )

    with h5py.File(output_h5_file, "w") as hf:
        hf.create_dataset("fids", data=data)
        hf.attrs["file_created_date"] = (
            datetime.now().isoformat().replace("+00:00", "Z")
        )
        try:
            hf.attrs["file_created_by"] = getpass.getuser()
        except (OSError, KeyError):
            hf.attrs["file_created_by"] = "unknown_user"

    # Check if file exists
    if not output_h5_file.exists():
        raise FileNotFoundError(
            f"Output file {output_h5_file} was not created successfully."
        )

    return data, output_h5_file


def _loadmat(filename: str | Path) -> dict:
    """
    Load a MATLAB .mat file and convert its contents into Python data structures.

    This function wraps around `scipy.io.loadmat` to address issues with
    recovering Python dictionaries from MATLAB .mat files. It ensures that
    all entries, including those stored as MATLAB objects, are properly
    converted to Python-native types by invoking the `_check_keys` function.

    Args:
        filename (str): The path to the .mat file to be loaded.

    Returns
    -------
        dict: A dictionary containing the contents of the .mat file, with
        MATLAB objects converted to Python-native types.
    """
    import scipy.io

    data = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(data: dict) -> dict:
    """Recursively convert MATLAB structs within a dictionary to Python dictionaries.

    Iterates through the keys of the given dictionary and checks if any of the values
    are MATLAB structs (from `scipy.io.matlab.mat_struct`). If a MATLAB struct is
    found, it is converted to a Python dictionary using the `_todict` function.

    Parameters
    ----------
    data : dict
        The input dictionary to process.

    Returns
    -------
    dict
        The processed dictionary with MATLAB structs converted to Python dictionaries.
    """
    from scipy.io.matlab import mat_struct

    for key in data:
        if isinstance(data[key], mat_struct):
            data[key] = _todict(data[key], mat_struct)
    return data


def _todict(matobj, mat_struct_type: type) -> dict:
    """Recursively construct nested dictionaries from a MATLAB struct object."""
    out: dict = {}
    for field_name in matobj._fieldnames:
        elem = matobj.__dict__[field_name]
        if isinstance(elem, mat_struct_type):
            out[field_name] = _todict(elem, mat_struct_type)
        else:
            out[field_name] = elem
    return out
