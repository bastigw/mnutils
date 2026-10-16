import getpass
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from loguru import logger

from . import file_helpers

#: The ``h`` header fields (``"struct/field"`` paths) MNUtils reads from a ``ScanArchive*.mat``.
#: ``RawMRISeries`` header properties read 26 of these; ``spectral_width`` is the one extra field
#: ``pyAMARES.fileio.readfidall.header2par_v73`` needs. A v7.3 header holds ~1600 HDF5 datasets and
#: reading them all costs ~0.45 s per file, so ``read_mat_header`` only touches these. A new header
#: property must be listed here (``tests/test_data_loaders.py`` enforces it).
HEADER_FIELDS: tuple[str, ...] = (
    "rdb_hdr/data_collect_type",
    "rdb_hdr/run_int",
    "rdb_hdr/scan_date",
    "rdb_hdr/scan_time",
    "rdb_hdr/nframes",
    "rdb_hdr/user0",
    "rdb_hdr/user3",
    "rdb_hdr/user14",
    "rdb_hdr/user25",
    "rdb_hdr/ps_mps_freq",
    "rdb_hdr/ps_mps_r1",
    "rdb_hdr/ps_mps_r2",
    "rdb_hdr/ps_mps_tg",
    "rdb_hdr/te",
    "rdb_hdr/fov",
    "rdb_hdr/recon",
    "rdb_hdr/spectral_width",
    "image/specnuc",
    "image/mr_flip",
    "image/nex",
    "image/tr",
    "image/slthick",
    "image/sctime",
    "series/prtcl",
    "series/se_desc",
    "exam/ex_no",
    "mrconfig/fieldStrength",
)


def load_mat_file(file_path: Path | str, variable_names: Sequence[str] | None = None) -> dict:
    """Load a .mat file and return its contents as a dictionary.

    Transparently falls back from `scipy.io.loadmat` (plain v5/v7 files) to
    `mat73` (v7.3, HDF5-based files) when the former can't read the file.

    Parameters
    ----------
    file_path : Path or str
        Path to the .mat file.
    variable_names : sequence of str, optional
        Top-level variables to load; everything else is skipped. Defaults to all of them. On a
        v7.3 file this matters: the scanner header ``h`` alone is ~1600 HDF5 datasets.

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
        return _loadmat(file_path, variable_names)
    except (NotImplementedError, ValueError):
        # If it fails, try loading with mat73 (for v7.3 .mat files)
        try:
            import mat73

            if variable_names is not None:
                # Not `mat73.loadmat(only_include=...)`: it matches by substring, so asking for
                # everything but ``h`` still loads ``h`` whenever a variable is named ``hz``.
                return _unpack_v73(file_path, variable_names)
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


def list_mat_variables(file_path: Path | str) -> list[str]:
    """List a .mat file's top-level variable names without loading any of them.

    Parameters
    ----------
    file_path : Path or str
        Path to the .mat file (v5/v7 or v7.3).

    Returns
    -------
    list of str
        The variable names (MATLAB's internal ``#refs#``/``#subsystem#`` groups excluded).
    """
    if h5py.is_hdf5(file_path):
        with h5py.File(file_path, "r") as hf:
            return [name for name in hf if name not in ("#refs#", "#subsystem#")]
    import scipy.io

    return [name for name, _shape, _class in scipy.io.whosmat(file_path)]


def read_mat_header(file_path: Path | str, fields: Sequence[str] = HEADER_FIELDS) -> dict:
    """Read selected fields of a ``ScanArchive*.mat``'s ``h`` header, without loading the rest.

    Returns the same nested dict as ``load_mat_file(file_path)["h"]``, pruned to ``fields``.
    On a v7.3 file only those datasets are opened (~6 ms instead of ~450 ms); on a plain v5/v7
    file ``h`` is loaded alone and pruned. Fields missing from the file are left out.

    Parameters
    ----------
    file_path : Path or str
        Path to the .mat file.
    fields : sequence of str, optional
        ``"struct/field"`` paths below ``h``. Defaults to `HEADER_FIELDS`.

    Returns
    -------
    dict
        Nested dict (``{"series": {"se_desc": ...}}``); empty if the file has no ``h``.

    Raises
    ------
    FileNotFoundError
        If `file_path` does not exist or is not a file.
    ValueError
        If `file_path` does not have a `.mat` extension.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(
            f"The specified path does not exist or is not a file: '{file_path}'"
        )
    if file_path.suffix.lower() != ".mat":
        raise ValueError(f"The file does not have a .mat extension: '{file_path}'")

    if not h5py.is_hdf5(file_path):
        full = load_mat_file(file_path, variable_names=["h"]).get("h", {})
        header: dict = {}
        for field in fields:
            value = full
            try:
                for key in field.split("/"):
                    value = value[key]
            except (KeyError, TypeError):
                continue
            _set_nested(header, field, value)
        return header

    # Reuse mat73's converter per dataset so values match `load_mat_file` exactly (char arrays
    # become str, 1x1 doubles scalars, ...), but skip its walk over every dataset in the group.
    from mat73 import HDF5Decoder

    decoder = HDF5Decoder(verbose=False)
    header = {}
    with h5py.File(file_path, "r") as hf:
        if "h" not in hf:
            return header
        if "#refs#" in hf:
            decoder.refs = hf["#refs#"]
        group = hf["h"]
        for field in fields:
            if field in group:
                _set_nested(header, field, decoder.unpack_mat(group[field]))
    return header


def _unpack_v73(file_path: Path, names: Sequence[str]) -> dict:
    """Decode the named top-level variables of a v7.3 file with mat73's converter.

    Names the file lacks are skipped. Unlike ``mat73.loadmat(only_include=...)`` the selection is
    exact, not a substring match.
    """
    from mat73 import HDF5Decoder

    decoder = HDF5Decoder()
    with h5py.File(file_path, "r") as hf:
        if "#refs#" in hf:
            decoder.refs = hf["#refs#"]
        return {name: decoder.unpack_mat(hf[name]) for name in names if name in hf}


def _set_nested(target: dict, path: str, value: object) -> None:
    """Set ``target["a"]["b"] = value`` for ``path == "a/b"``, creating the inner dicts."""
    *parents, leaf = path.split("/")
    for key in parents:
        target = target.setdefault(key, {})
    target[leaf] = value


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
            logger.debug(f"Found existing raw FIDs file: {output_h5_file}. Loading data from it.")
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
            logger.warning(f"Output file {output_h5_file} already exists. Returning data from it.")
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
        raise ValueError(f"Loaded data is not a complex numpy array. Got type: {type(data)}")

    with h5py.File(output_h5_file, "w") as hf:
        hf.create_dataset("fids", data=data)
        hf.attrs["file_created_date"] = datetime.now().isoformat().replace("+00:00", "Z")
        try:
            hf.attrs["file_created_by"] = getpass.getuser()
        except (OSError, KeyError):
            hf.attrs["file_created_by"] = "unknown_user"

    # Check if file exists
    if not output_h5_file.exists():
        raise FileNotFoundError(f"Output file {output_h5_file} was not created successfully.")

    return data, output_h5_file


def _loadmat(filename: str | Path, variable_names: Sequence[str] | None = None) -> dict:
    """
    Load a MATLAB .mat file and convert its contents into Python data structures.

    This function wraps around `scipy.io.loadmat` to address issues with
    recovering Python dictionaries from MATLAB .mat files. It ensures that
    all entries, including those stored as MATLAB objects, are properly
    converted to Python-native types by invoking the `_check_keys` function.

    Args:
        filename (str): The path to the .mat file to be loaded.
        variable_names (sequence of str, optional): Only load these top-level variables.

    Returns
    -------
        dict: A dictionary containing the contents of the .mat file, with
        MATLAB objects converted to Python-native types.
    """
    import scipy.io

    data = scipy.io.loadmat(
        filename,
        struct_as_record=False,
        squeeze_me=True,
        variable_names=None if variable_names is None else list(variable_names),
    )
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
