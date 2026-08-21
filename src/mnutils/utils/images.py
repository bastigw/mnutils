import os
from pathlib import Path
from typing import TYPE_CHECKING, cast, overload

# This block is ignored at runtime. I only want to import later to reduce overhead.
if TYPE_CHECKING:
    import HD_BET.checkpoint_download
    import HD_BET.hd_bet_prediction
    import torch

from loguru import logger
from pandas.compat._optional import import_optional_dependency

from . import file_helpers


@overload
def extract_brain(
    file: str | Path,
    device: str = "cuda",
    use_tta: bool = True,
    save_mask: bool = True,
    extract_brain: bool = True,
    verbose: bool = False,
) -> Path: ...


@overload
def extract_brain(
    file: list[str | Path],
    device: str = "cuda",
    use_tta: bool = True,
    save_mask: bool = True,
    extract_brain: bool = True,
    verbose: bool = False,
) -> list[Path]: ...


def extract_brain(
    file: str | Path | list[str | Path],
    device: str = "cuda",
    use_tta: bool = True,
    save_mask: bool = True,
    extract_brain: bool = True,
    verbose: bool = False,
) -> Path | list[Path]:
    """Extract the brain from an image file using HD-BET.

    This function processes NIfTI files or directories containing DICOM files to extract the
    brain region using the HD-BET (HD-Brain Extraction Tool). It supports optional test-time
    augmentation (TTA) and allows saving the brain mask and/or the brain-extracted image.

    Parameters
    ----------
    file : str | Path | list[str | Path]
        The path(s) to the image file(s) or directories containing DICOM files to extract the
        brain from. If a directory is provided, it will be converted to a NIfTI file first.
    device : str, optional
        The device to use for processing. Default is "cuda".
    use_tta : bool, optional
        Whether to use test-time augmentation (TTA) during brain extraction. Default is True.
    save_mask : bool, optional
        Whether to save the brain mask. Default is True.
    extract_brain : bool, optional
        Whether to compute and save the brain-extracted image. Default is True.
    verbose : bool, optional
        Whether to enable verbose logging. Default is False.

    Returns
    -------
    Path | list[Path]
        The path(s) to the brain-extracted NIfTI file(s). If a single file is provided as input,
        a single Path object is returned. If a list of files is provided, a list of Path objects
        is returned.

    Raises
    ------
    ValueError
        If the input file does not have a ".nii.gz" suffix or if the input is invalid.
    ImportError
        If the required dependencies (torch, HD-BET) are not installed.

    Notes
    -----
    - This function requires the HD-BET library and PyTorch to be installed.
    - If the input is a directory containing DICOM files, it will be converted to a NIfTI file
      before processing.
    - The function checks for existing brain-extracted files in the output directory to avoid
      overwriting existing results.
    - The HD-BET model parameters will be downloaded automatically if not already available.

    Examples
    --------
    Extract the brain from a single NIfTI file:
    >>> extract_brain("path/to/image.nii.gz")
    Extract the brain from multiple NIfTI files:
    >>> extract_brain(["path/to/image1.nii.gz", "path/to/image2.nii.gz"])
    Extract the brain from a directory containing DICOM files:
    >>> extract_brain("path/to/dicom_directory")

    """
    # If the the input is a folder containing dicoms first convert to nifti
    if isinstance(file, (str, Path)):
        single_file = True
        file = [Path(file)]
    else:
        single_file = False

    files: list[Path] = [Path(f) for f in file]

    for idx, input_path in enumerate(files):
        if os.path.isdir(input_path):
            nifti_path = file_helpers.get_nifti_data(input_path, convert_dicoms=True)
            # Replace the original path with the new nifti path
            files[idx] = nifti_path

    # type hinting
    torch = import_optional_dependency("torch")
    torch = cast("torch", torch)
    hd_bet_checkpoint_download = import_optional_dependency("HD_BET.checkpoint_download")
    hd_bet_checkpoint_download = cast("HD_BET.checkpoint_download", hd_bet_checkpoint_download)
    hd_bet_predictor = import_optional_dependency("HD_BET.hd_bet_prediction")
    hd_bet_predictor = cast("HD_BET.hd_bet_prediction", hd_bet_predictor)

    # Start running hd_bet
    # TODO fix issues with stdout capture

    downloaded_parameters = False
    predictor = None

    output_files: list[Path] = []

    for idx, input_path in enumerate(files):
        logger.debug(
            f"Processing file {file_helpers._relative_to_cwd(input_path)} ({idx + 1}/{len(files)})"
        )
        # Assert that input path has nii.gz suffix
        if not str(input_path).endswith(".nii.gz"):
            raise ValueError(
                f"Input file {input_path} does not have a .nii.gz suffix. "
                "Please provide a valid NIfTI file."
            )

        output_dir = Path(input_path).parent
        default_output_path = (
            output_dir / f"{str(input_path.name)[:-7]}_bet.nii.gz"
        )  # Bet will be saved with _bet suffix by default
        other_bet_niftis = file_helpers.get_nifti_file(
            input_path.parent,
            filter_out_keywords=[],
            filter_in_keywords=["_bet.nii.gz"],
        )

        # Check if output file already exists
        if default_output_path.exists():
            logger.warning(
                f"Output file {file_helpers._relative_to_cwd(default_output_path)} "
                f"already exists. Skipping brain extraction for "
                f"{file_helpers._relative_to_cwd(input_path)}."
            )
            output_files.append(default_output_path)
            continue

        if other_bet_niftis:
            logger.warning(
                f"Other brain extracted NIfTI files found in output directory "
                f"{file_helpers._relative_to_cwd(output_dir)}. This may indicate that "
                f"brain extraction has already been performed for this or other files. Skipping "
                f"brain extraction for {file_helpers._relative_to_cwd(input_path)} to "
                f"avoid overwriting existing files."
            )
            output_files.append(other_bet_niftis)
            continue

        if not downloaded_parameters:
            hd_bet_checkpoint_download.maybe_download_parameters()
            downloaded_parameters = True

        if predictor is None:
            predictor = hd_bet_predictor.get_hdbet_predictor(
                use_tta=use_tta, device=torch.device(device), verbose=verbose
            )

        hd_bet_predictor.hdbet_predict(
            str(input_path),
            str(input_path),
            predictor,
            keep_brain_mask=save_mask,
            compute_brain_extracted_image=extract_brain,
        )

        logger.debug(
            f"Brain extracted successfully for "
            f"{file_helpers._relative_to_cwd(input_path)}, saved to "
            f"{file_helpers._relative_to_cwd(default_output_path)}"
        )
        output_files.append(default_output_path)

    if single_file:
        return output_files[0]
    else:
        return output_files
