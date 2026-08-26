import os
import re
from datetime import datetime
from pathlib import Path

import h5py
import pandas as pd
from IPython.core.getipython import get_ipython
from IPython.display import display
from loguru import logger

_DEFAULT_DICOM_FOLDER_PATTERN = r"(?:Series)?(\d{1,5})_"


def _relative_to_cwd(path: str | Path) -> Path:
    """Express a path relative to the current working directory, for log messages.

    Falls back to the unchanged path when no relative form exists -- notably on
    Windows, where a path on another drive (e.g. a ``C:`` temp dir while the cwd
    is on ``D:``) has a different anchor and ``Path.relative_to`` raises.

    Parameters
    ----------
    path : str or Path
        The path to shorten.

    Returns
    -------
    Path
        The path relative to the cwd, or the original path if that is impossible.
    """
    path = Path(path)
    try:
        return path.relative_to(os.getcwd(), walk_up=True)
    except ValueError:
        return path


def get_all_dicom_series_ids(data_folder: str | Path) -> tuple[list[int], list[int]]:
    """Retrieve all DICOM series IDs from a specified folder.

    Scans the given folder for subdirectories whose names match the pattern
    "0XX", where "XX" is a one- or two-digit number. Not including series
    with three digits.

    Parameters
    ----------
    data_folder : str or Path
        The path to the folder containing the DICOM series in folders.

    Returns
    -------
    tuple[list[int], list[int]]
        A sorted list of integers representing the DICOM series IDs, and a
        list of missing series IDs, if any.
    """
    data_folder = Path(data_folder)
    pattern = re.compile(_DEFAULT_DICOM_FOLDER_PATTERN)
    series_numbers = []
    for folder in data_folder.iterdir():
        match = pattern.match(folder.name)
        if match and folder.is_dir():
            series_id = int(match.group(1))
            series_numbers.append(series_id)

    series_numbers.sort()
    logger.debug(f"Found series IDs: {series_numbers}")

    # Only flag gaps within the low, dense protocol-step range (series <= 99).
    # Series >= 100 are reformats/resaves with sparse-by-design numbering
    # (e.g. 500/501, 40003) and must never be flagged as "missing" -- nor may
    # they stretch the checked range: it ends at the last series <= 99 that
    # actually exists, not at 99.
    dense = [n for n in series_numbers if n <= 99]
    missing = []
    if dense:
        missing = sorted(set(range(dense[0], dense[-1] + 1)) - set(dense))
    if missing:
        logger.error(f"Missing series IDs: {missing}. DOUBLE CHECK DATA FOLDER!")

    return series_numbers, missing


def get_all_exam_series_ids(exam_folder: str | Path) -> list[int]:
    """Retrieve all exam series IDs from a specified folder.

    Scans the given folder for subdirectories whose names match the pattern
    "SeriesX" or "SeriesXX", where X represents a digit.

    Parameters
    ----------
    exam_folder : str or Path
        The path to the folder containing the exam series.

    Returns
    -------
    list[int]
        A sorted list of integers representing the series IDs.
    """
    exam_folder = Path(exam_folder)
    series_folders = [
        folder
        for folder in exam_folder.iterdir()
        if folder.is_dir() and re.match(r"Series\d{1,2}", folder.name)
    ]
    series_numbers = [
        int(re.search(r"\d{1,2}", folder.name).group())  # type: ignore
        for folder in series_folders
    ]
    series_numbers.sort()
    return series_numbers


def get_exam_folder(base_path: str | Path) -> Path:
    """Find a folder containing 'Exam' in its name within the specified base path.

    Parameters
    ----------
    base_path : str or Path
        The base directory to search.

    Returns
    -------
    Path
        The Path object to the found 'Exam' folder.

    Raises
    ------
    ValueError
        If no 'Exam' folder is found or if multiple 'Exam' folders exist.
    """
    base = Path(base_path)
    if not base.is_dir():
        raise ValueError(f"The base path does not exist or is not a folder: '{base_path}'")

    exam_folders = [item for item in base.iterdir() if "Exam" in item.name and item.is_dir()]

    if len(exam_folders) == 0:
        raise ValueError(f"No folder with 'Exam' in its name found in '{base_path}'")
    elif len(exam_folders) > 1:
        raise ValueError(f"Multiple folders with 'Exam' in their name found in '{base_path}'")

    return exam_folders[0]


def get_mat_data_from_series(base_folder: str | Path, series: int) -> Path:
    """Find a specific .mat file within a designated series folder.

    Parameters
    ----------
    base_folder : str or Path
        The base directory.
    series : int
        The series number.

    Returns
    -------
    Path
        The Path object to the first found .mat file.

    Raises
    ------
    ValueError
        If `base_folder` does not exist or `series` is not a positive integer.
    FileNotFoundError
        If no .mat files are found in the series folder.
    """
    base = Path(base_folder)
    if not base.is_dir():
        raise ValueError(f"The base folder does not exist: '{base_folder}'")
    if not isinstance(series, int) or series <= 0:
        raise ValueError("Series must be a positive integer.")

    exam_folder = get_exam_folder(base)
    series_folder = exam_folder / f"Series{series}"

    mat_files = list(series_folder.glob("ScanArchive*.mat"))

    if not mat_files:
        raise FileNotFoundError(
            f"No .mat files found in the Series folder.\nFolder path: '{series_folder}'"
        )
    elif len(mat_files) == 1:
        return mat_files[0]
    else:
        logger.warning(
            "Multiple .mat files found in the Series folder. Selecting the first one.\n"
            f"File '{mat_files[0].name}', Folder path: '{series_folder}'"
        )
        return mat_files[0]


def get_h5_data_from_series(
    base_folder: str | Path, series: int, file_start: str = "ScanArchive"
) -> Path:
    """Find a specific .h5 file within a designated series folder.

    Parameters
    ----------
    base_folder : str or Path
        The base directory.
    series : int
        The series number.
    file_start : str, optional
        The filename prefix to match. Defaults to "ScanArchive".

    Returns
    -------
    Path
        The Path object to the first found .h5 file.

    Raises
    ------
    ValueError
        If `base_folder` does not exist or `series` is not a positive integer.
    FileNotFoundError
        If no .h5 files are found in the series folder.
    """
    base = Path(base_folder)
    if not base.is_dir():
        raise ValueError(f"The base folder does not exist: '{base_folder}'")
    if not isinstance(series, int) or series <= 0:
        raise ValueError("Series must be a positive integer.")

    exam_folder = get_exam_folder(base)
    series_folder = exam_folder / f"Series{series}"

    h5_files = list(series_folder.glob(f"{file_start}*.h5"))

    if not h5_files:
        raise FileNotFoundError(
            f"No .h5 files found in the Series folder.\nFolder path: '{series_folder}'"
        )
    elif len(h5_files) == 1:
        return h5_files[0]
    else:
        logger.warning(
            "Multiple .h5 files found in the Series folder. Selecting the first one.\n"
            f"File '{h5_files[0].name}', Folder path: '{series_folder}'"
        )
        return h5_files[0]


def get_latest_processing_file(output_folder: Path, series_id: int) -> Path:
    """Get the latest processing file for a given series ID.

    Parameters
    ----------
    output_folder : Path
        The output folder where processing files are stored.
    series_id : int
        The series ID to look for.

    Returns
    -------
    Path
        The path to the latest processing file.

    Raises
    ------
    FileNotFoundError
        If no processing files are found for the given series ID.
    """
    processing_files = list(output_folder.glob(f"*Series{series_id:02d}_processing_data.h5"))

    if not processing_files:
        raise FileNotFoundError(
            f"No processing files found for series ID {series_id} in {output_folder}"
        )

    # Sort files by modification time and return the latest one
    latest_file = max(processing_files, key=os.path.getmtime)
    return latest_file


def get_dicom_folder(base_folder: str | Path, series: int) -> Path:
    """Find a specific DICOM folder within a designated series folder.

    Parameters
    ----------
    base_folder : str or Path
        The base directory.
    series : int
        The series number.

    Returns
    -------
    Path
        The Path object to the DICOM folder.

    Raises
    ------
    ValueError
        If `base_folder` does not exist, or no (or multiple) DICOM folders
        match `series`.
    KeyError
        If `series` is not a positive integer.
    """
    base = Path(base_folder)
    if not base.is_dir():
        raise ValueError(f"The base folder does not exist: '{base_folder}'")
    if not isinstance(series, int) or series <= 0:
        raise KeyError("Series must be a positive integer.")

    # Compile the regex pattern
    pattern = re.compile(_DEFAULT_DICOM_FOLDER_PATTERN)

    # Look for a folder that matches the series number using the regex pattern
    dicom_folders = []
    for item in base.iterdir():
        if item.is_dir():
            match = pattern.match(item.name)
            if match and int(match.group(1)) == series:
                dicom_folders.append(item)

    if len(dicom_folders) == 0:
        raise ValueError(
            f"No DICOM folder matching the series number '{series}' found in '{base_folder}'. "
            f"Ensure the folder name follows the pattern '{_DEFAULT_DICOM_FOLDER_PATTERN}'."
        )
    elif len(dicom_folders) > 1:
        raise ValueError(
            f"Multiple DICOM folders matching the series number '{series}' found "
            f"in '{base_folder}'. Check for duplicate folders or naming inconsistencies."
        )
    else:
        # For later processing its important that the folder name does not contain whitespaces
        # Rename the folder to the same with all whitespaces replaced by underscores
        dicom_folder = dicom_folders[0]
        new_name = dicom_folder.name.replace(" ", "_")
        if new_name != dicom_folder.name:
            new_path = dicom_folder.parent / new_name
            dicom_folder.rename(new_path)
            dicom_folder = new_path
        return dicom_folder


def get_nifti_file(
    folder: Path,
    filter_out_keywords: list[str] | None = None,
    filter_in_keywords: list[str] | None = None,
) -> Path | None:
    """Find the first NIfTI file in a folder matching the given keyword filters.

    Parameters
    ----------
    folder : Path
        The folder to search for NIfTI files.
    filter_out_keywords : list[str] or None, optional
        Keywords whose presence in a filename excludes it. Defaults to
        ["bet", "skullstrip", "ss", "resampled"] when None.
    filter_in_keywords : list[str] or None, optional
        Keywords a filename must contain to be included. Defaults to no
        filtering when None.

    Returns
    -------
    Path or None
        The first matching NIfTI file, or None if no file matches.
    """
    if filter_out_keywords is None:
        filter_out_keywords = ["bet", "skullstrip", "ss", "resampled"]
    if filter_in_keywords is None:
        filter_in_keywords = []

    # Get all nifti files of the folder
    # Make sure the filename does not contain "bet" or "skullstrip" or "ss"
    all_nifti_files = list(folder.glob("*.nii")) + list(folder.glob("*.nii.gz"))
    # Make sure to filter out files that contain "bet", "skullstrip", or "ss" in their names
    filter_out_regex = (
        re.compile("|".join(filter_out_keywords), re.IGNORECASE) if filter_out_keywords else None
    )
    filter_in_regex = (
        re.compile("|".join(filter_in_keywords), re.IGNORECASE) if filter_in_keywords else None
    )

    logger.trace(
        f"Filtering NIfTI files in '{folder}' with keywords not in word: "
        f"{filter_out_keywords} and keywords in word: {filter_in_keywords}"
    )
    filtered_nifti_files = all_nifti_files
    if filter_out_regex:
        logger.trace(f"Filter out regex: {filter_out_regex.pattern}")
        filtered_nifti_files = [
            f for f in filtered_nifti_files if not filter_out_regex.search(f.name)
        ]
    if filter_in_regex:
        filtered_nifti_files = [f for f in filtered_nifti_files if filter_in_regex.search(f.name)]

    selected_file = filtered_nifti_files[0] if filtered_nifti_files else None
    logger.trace(f"Selected NIfTI file: {selected_file}")
    return selected_file


def get_niftis_from_series(
    base_folder: str | Path, series: int, convert_dicoms: bool = False, **kwargs
) -> Path:
    """Find a specific NIfTI file within a designated series folder.

    Parameters
    ----------
    base_folder : str or Path
        The base directory.
    series : int
        The series number.
    convert_dicoms : bool, optional
        If True, converts DICOM files to NIfTI format if no NIfTI files are
        found. Defaults to False.
    **kwargs
        Additional keyword arguments passed to `get_nifti_data`.

    Returns
    -------
    Path
        The Path object to the first found NIfTI file.

    Raises
    ------
    FileNotFoundError
        If no NIfTI files are found in the series folder and `convert_dicoms`
        is False (or conversion produced no NIfTI files).
    """
    series_folder = get_dicom_folder(base_folder, series)
    return get_nifti_data(series_folder, convert_dicoms=convert_dicoms, **kwargs)


def get_nifti_data(folder: str | Path, convert_dicoms: bool = False, **kwargs) -> Path:
    """
    Retrieve NIfTI files from a folder, optionally converting DICOM files to NIfTI format.

    This function searches for NIfTI files in the specified folder. If no NIfTI files are found
    and convert_dicoms is True, it attempts to convert DICOM files in the folder to NIfTI format.

    Parameters
    ----------
    folder : str | Path
        The folder path to search for NIfTI files or containing DICOM files to convert.
    convert_dicoms : bool, optional
        If True, attempt to convert DICOM files to NIfTI format if no NIfTI files are found.
        Default is False.
    **kwargs
        Additional keyword arguments to pass to get_nifti_file().

    Returns
    -------
    Path
        The path to a NIfTI file found in the folder.

    Raises
    ------
    FileNotFoundError
        If no NIfTI files are found in the folder and either:
        - convert_dicoms is False, or
        - convert_dicoms is True but the conversion attempt did not produce any NIfTI files.

    Notes
    -----
    - The function first attempts to find existing NIfTI files without conversion.
    - If convert_dicoms is True and no NIfTI files are found, DICOM files are converted
      using dcm2niiw with the filename format "%s_%d_%a".
    - The dcm2niiw package must be installed to use the convert_dicoms functionality.

    Examples
    --------
    >>> nifti_path = get_niftis('/path/to/series/folder')
    >>> nifti_path = get_niftis('/path/to/dicom/folder', convert_dicoms=True)


    """
    folder = Path(folder)

    if nifti_file := get_nifti_file(folder, **kwargs):
        return nifti_file

    if convert_dicoms:
        # Check if dcm2niiw is available else return with error and recommend installing it
        from dcm2niiw import dcm2nii

        dcm2nii(
            in_folder=folder,
            out_folder=folder,
            filename_format="%s_%d_%a",
        )

        if nifti_file := get_nifti_file(folder, **kwargs):
            return nifti_file

        raise FileNotFoundError(
            "No NIfTI files found in the Series folder after conversion attempt.\n"
            f"Folder path: '{folder}'"
        )

    error_msg = f"No NIfTI files found in the Series folder.\nFolder path: '{folder}'"
    if not convert_dicoms:
        error_msg += (
            "\nYou may try running with `convert_dicoms=True` to convert DICOM files "
            "to NIfTI format."
        )
    raise FileNotFoundError(error_msg)


def print_hdf5_tree(hdf5_file: h5py.File | h5py.Group, prefix: str = "") -> str:
    """Return a string containing the tree representation of the HDF5 file.

    Parameters
    ----------
    hdf5_file : h5py.File or h5py.Group
        The HDF5 file object.
    prefix : str, optional
        The line prefix, used for recursive indentation. Defaults to "".

    Returns
    -------
    str
        The full or partial tree representation of the file.
    """
    tree_string = ""
    items_index = len(hdf5_file)

    for key, hdf5_value in hdf5_file.items():
        items_index -= 1

        branch_symbol = "├──"
        prefix_symbol = "|"

        if items_index == 0:
            branch_symbol = "└──"
            prefix_symbol = " "

        if isinstance(hdf5_value, h5py.Group):
            tree_string += f"{prefix}{branch_symbol} {key}\n"
            tree_string += print_hdf5_tree(hdf5_value, f"{prefix}{prefix_symbol}   ")

        else:
            try:
                tree_string += f"{prefix}{branch_symbol} "
                if hdf5_value.shape == ():
                    tree_string += f"{key} {str(hdf5_value[()])}\n"
                else:
                    tree_string += f"{key} {hdf5_value.shape}\n"

            except TypeError:
                tree_string += f"{prefix}{branch_symbol} {key} (scalar)\n"

    return tree_string


def get_exam_overview(data_folder: str | Path, print_overview: bool = True) -> pd.DataFrame:
    """Build (and optionally print) an overview of all exam series in a data folder.

    Parameters
    ----------
    data_folder : str or Path
        The path to the folder containing the DICOM series in folders.
    print_overview : bool, optional
        Whether to print/display the overview. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the overview of all exam series, indexed by
        series ID, with "Has Exam", "Has DICOM" and "Folder Name" columns.
    """
    series_ids, missing = get_all_dicom_series_ids(data_folder)
    try:
        exam_folder = get_exam_folder(data_folder)
        exam_ids = get_all_exam_series_ids(exam_folder)
    except ValueError:
        logger.warning("No exam folder found. Skipping exam overview.")
        exam_folder = None
        exam_ids = []

    overview_data = []

    all_ids = sorted(set(series_ids).union(exam_ids))
    for series_id in all_ids:
        try:
            dicom_folder = get_dicom_folder(data_folder, series_id)
            folder_name = dicom_folder.name
            has_dicom = True
        except ValueError:
            folder_name = None
            has_dicom = False

        overview_data.append(
            {
                "Series ID": series_id,
                "Has Exam": series_id in exam_ids,
                "Has DICOM": has_dicom,
                "Folder Name": folder_name,
            }
        )

    overview_df = pd.DataFrame(overview_data).set_index("Series ID")

    if print_overview:
        if is_running_in_jupyter():

            def highlight_rows(row):
                if not row["Has DICOM"]:
                    return ["background-color: lightsalmon"] * len(row)
                elif row["Has Exam"]:
                    return ["background-color: lightblue"] * len(row)
                return [""] * len(row)

            styled_df = overview_df.style.apply(highlight_rows, axis=1)
            display(styled_df)
            # Print info about the color coding
            logger.debug("Color coding: blue = Has Exam Data, orange = Missing DICOM")
            # Log exam path
            logger.debug(f"Exam folder path: {exam_folder}")
        else:
            logger.info(f"\n{overview_df.to_string()}")

    return overview_df


def is_running_in_jupyter() -> bool:
    """Check if the code is being run within an IPython kernel (Jupyter Notebook or Console)."""
    try:
        # Check if the function exists
        shell = get_ipython().__class__.__name__

        if shell == "ZMQInteractiveShell":
            # This is the shell used by Jupyter notebooks/labs
            return True
        elif shell == "TerminalInteractiveShell":
            # This is the shell used by the IPython terminal (not a notebook)
            return False
        else:
            # Other interactive shells
            return False

    except NameError:
        # get_ipython is not defined, so it's a standard Python interpreter/script
        return False


def move_files_with_glob(
    folder: str | Path,
    file_glob: str,
    target_folder_name: str = "old",
    prepend_creation_date: bool = True,
) -> None:
    """Move files matching a glob pattern from a folder to a target folder.

    Parameters
    ----------
    folder : str or Path
        The folder to search for files.
    file_glob : str
        The glob pattern to look for (e.g. '*.csv'), passed directly to
        `Path.glob`.
    target_folder_name : str, optional
        The name of the target folder, created inside `folder`. Defaults to
        "old".
    prepend_creation_date : bool, optional
        Whether to prepend the creation date to the filename when moving.
        Defaults to True.

    Raises
    ------
    ValueError
        If the folder does not exist or is not a directory.
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"The specified folder does not exist or is not a directory: '{folder}'")

    target_folder = folder / target_folder_name
    # Check if there are any files matching the glob pattern
    files_to_move = list(folder.glob(file_glob))
    if not files_to_move:
        logger.debug(f"No files found matching the pattern '{file_glob}' in '{folder}'.")
        return

    # Create the target folder only if there are files to move
    target_folder.mkdir(exist_ok=True)

    for file in folder.glob(file_glob):
        target_file = target_folder / file.name
        # Check if target file already exists
        target_file_exists = target_file.exists()
        if target_file_exists:
            logger.warning(
                f"File '{target_file}' already exists. Renaming the existing file "
                "by prepending its creation date."
            )
            # Rename the existing file in the target folder by prepending its creation date
            creation_time = target_file.stat().st_mtime
            date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
            timestamped_existing_file = f"{date_str}_{target_file.name}"
            target_file.rename(target_file.parent / timestamped_existing_file)
            logger.debug(f"Renamed existing file to: {timestamped_existing_file}")

        if prepend_creation_date:
            # Before prepending date check if the file already has a date prefix, if so, do not
            # prepend another one
            if re.match(r"^\d{8}_\d{6}_", file.name):
                logger.debug(
                    f"File '{file.name}' already has a date prefix. Moving without "
                    "prepending another date."
                )
            else:
                creation_time = file.stat().st_mtime
                date_str = datetime.fromtimestamp(creation_time).strftime("%Y%m%d_%H%M%S")
                new_name = f"{date_str}_{file.name}"
                target_file = target_folder / new_name

        file.rename(target_file)
        if target_file.is_absolute():
            logger.debug(f"Moved file to: {_relative_to_cwd(target_file)}")
        else:
            logger.debug(f"Moved file to: {target_file}")
