from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import nibabel as nib
import numpy as np
import numpy.typing as npt
from loguru import logger
from nibabel import processing, spatialimages

from .. import GESeries

DEFAULT_PARAMS = {
    "orientation": ("L", "P", "S"),
}

DEFAULT_RESAMPLE_PARAMS = {
    "order": 0,
    "mode": "grid-constant",
    "cval": np.nan,
}

type DISPLAY_PLANES = Literal["axial", "coronal", "sagittal"]


def _resolve_orientation(
    orientation: tuple[str, str, str] | None, display_plane: DISPLAY_PLANES | None
) -> tuple[str, str, str]:
    """Resolve the target axis codes from `orientation`/`display_plane`.

    Shared by `orient_nifti` and `get_display_affine` so the two can't drift apart.
    """
    if display_plane is not None:
        match display_plane:
            case "axial":
                orientation = ("L", "P", "S")
            case "coronal":
                orientation = ("L", "I", "P")
            case "sagittal":
                orientation = ("I", "L", "P")
            case _:
                orientation = DEFAULT_PARAMS["orientation"]
        logger.debug(f"Orienting NIfTI to {display_plane} plane. Using orientation {orientation}.")
        return orientation

    if orientation is None:
        logger.debug(
            f"No orientation specified. Using default orientation: {DEFAULT_PARAMS['orientation']}."
        )
        return DEFAULT_PARAMS["orientation"]

    return orientation


def get_display_affine(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = None,
    display_plane: DISPLAY_PLANES | None = None,
) -> npt.NDArray[np.float64]:
    """Return the affine mapping display voxel indices (after orient_nifti) to world mm.

    orient_nifti applies apply_orientation then flipud(rot90(k=1)) which equals
    a transpose of axes 0 and 1. This function encodes both steps as an affine.
    Use inv(get_display_affine(t1)) @ mrsi_affine to get mrsi→display mapping.

    Parameters
    ----------
    nii : nibabel.spatialimages.SpatialImage or GESeries.NiiBase
        The image to compute the display affine for.
    orientation : tuple[str, str, str], optional
        Target axis codes. Defaults to `DEFAULT_PARAMS["orientation"]` if
        neither this nor `display_plane` is given.
    display_plane : {"axial", "coronal", "sagittal"}, optional
        If given, overrides `orientation` with a preset for the named plane,
        matching `orient_nifti`.
    """
    if isinstance(nii, GESeries.NiiBase):
        nii: spatialimages.SpatialImage = nii.nii
    orientation = _resolve_orientation(orientation, display_plane)
    # Reorient with the target axcodes applied directly (not composed with the
    # image's current orientation via ornt_transform) -- orient_nifti does the same on
    # the data via apply_orientation, and the two must use the identical permutation
    # or the zooms end up describing axes other than the ones actually displayed.
    target_ornt = nib.orientations.axcodes2ornt(orientation)
    reoriented = nii.as_reoriented(target_ornt)
    oriented_affine = np.asarray(reoriented.affine, dtype=float)
    # flipud(rot90(k=1, axes=(0,1))) = pure transpose of axes 0 and 1.
    # This swaps columns 0 and 1 of the affine.
    P = np.eye(4)
    P[0, 0] = P[1, 1] = 0.0
    P[0, 1] = P[1, 0] = 1.0
    return oriented_affine @ P


def get_display_zooms(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = None,
    display_plane: DISPLAY_PLANES | None = None,
) -> tuple[float, float]:
    """Return the (row, col) physical voxel edge lengths of the displayed grid, in mm.

    The column norms of `get_display_affine`'s first two columns give the
    physical spacing along the two in-plane axes `orient_nifti` puts at
    positions 0 and 1 -- correct even for oblique/rotated affines, unlike
    reading `nii.header.get_zooms()` and guessing which entries survived the
    reorientation/transpose. Meant to be passed straight through as
    `display_images(..., zooms=...)`.
    """
    display_affine = get_display_affine(nii, orientation=orientation, display_plane=display_plane)
    row_spacing = float(np.linalg.norm(display_affine[:3, 0]))
    col_spacing = float(np.linalg.norm(display_affine[:3, 1]))
    return row_spacing, col_spacing


def orient_nifti(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = None,
    display_plane: DISPLAY_PLANES | None = None,
    caching: bool = True,
) -> npt.NDArray[np.float64]:
    """Reorient and rotate a NIfTI image's data array for display.

    Applies the requested axis orientation (or one derived from
    `display_plane`), then transposes axes 0 and 1 to match the display
    convention used elsewhere in this module.

    Parameters
    ----------
    nii : spatialimages.SpatialImage or GESeries.NiiBase
        The image to orient.
    orientation : tuple[str, str, str], optional
        Target axis codes. Defaults to `DEFAULT_PARAMS["orientation"]` if
        neither this nor `display_plane` is given.
    display_plane : {"axial", "coronal", "sagittal"}, optional
        If given, overrides `orientation` with a preset for the named plane.
    caching : bool, optional
        Whether to cache the underlying image data (more memory, faster
        repeat access) or not (less memory, slower repeat access).
        Defaults to True.

    Returns
    -------
    ndarray
        The reoriented image data array.
    """
    if isinstance(nii, GESeries.NiiBase):
        nii: spatialimages.SpatialImage = nii.nii
    orientation = _resolve_orientation(orientation, display_plane)

    if caching:
        logger.debug(
            "Using caching for NIfTI data. This may use more memory but can speed up "
            "processing for repeat access."
        )
        caching_param = "fill"
    else:
        logger.debug(
            "Not using caching for NIfTI data. This may use less memory but can slow down "
            "processing for repeat access."
        )
        caching_param = "unchanged"

    imagesOr = nib.orientations.apply_orientation(
        nii.get_fdata(caching=caching_param),
        nib.orientations.axcodes2ornt(orientation),
    )
    images = np.flipud(np.rot90(imagesOr, k=1, axes=(0, 1)))
    return images


def apply_half_voxel_shift(
    affine: npt.NDArray[np.float64], shift_z_axis: bool = False
) -> npt.NDArray[np.float64]:
    """Shift an affine by half a voxel in x/y (and optionally z). Deprecated, always raises.

    Parameters
    ----------
    affine : ndarray
        The affine to shift.
    shift_z_axis : bool, optional
        Whether to also shift the z axis. Defaults to False.

    Returns
    -------
    ndarray
        The shifted affine. Unreachable, since this function always raises.

    Raises
    ------
    DeprecationWarning
        Always. Half voxel shifts are usually a symptom of a bad affine
        definition; use dcm2nii to convert DICOMs and use its affine instead.
    """
    raise DeprecationWarning(
        "This function should not be used as half voxel shifts are usually bad and a "
        "result of bad affine definition"
        "Use dcm2nii to convert dicoms to nifits and use the resulting affine"
    )
    # Half voxel shift should only be in x and y directions
    if shift_z_axis:
        dims = 3
    else:
        dims = 2

    shifted_affine = deepcopy(affine)
    shift = np.diag(affine[:dims, :dims]) * 0.5
    shifted_affine[:dims, 3] += shift
    logger.debug(f"Applied half voxel shift of {shift} to affine.")
    return shifted_affine


def resample_nifti(
    source_nii: spatialimages.SpatialImage | GESeries.NiiBase,
    target_nii: spatialimages.SpatialImage | GESeries.NiiBase,
    **kwargs: Any,
) -> spatialimages.SpatialImage:
    """Resample `source_nii` into `target_nii`'s voxel grid via `nibabel.processing`.

    Parameters
    ----------
    source_nii : spatialimages.SpatialImage or GESeries.NiiBase
        The image to resample.
    target_nii : spatialimages.SpatialImage or GESeries.NiiBase
        The image whose grid to resample onto.
    **kwargs
        Overrides for `DEFAULT_RESAMPLE_PARAMS`, passed to
        `nibabel.processing.resample_from_to`.

    Returns
    -------
    spatialimages.SpatialImage
        The resampled image.
    """
    if isinstance(source_nii, GESeries.NiiBase):
        source_nii: spatialimages.SpatialImage = source_nii.nii
    if isinstance(target_nii, GESeries.NiiBase):
        target_nii: spatialimages.SpatialImage = target_nii.nii
    # Resample data_nifti to base_nifti space
    resample_kwargs = DEFAULT_RESAMPLE_PARAMS.copy()
    resample_kwargs.update(kwargs)
    logger.debug(f"Resampling with parameters: {resample_kwargs}")

    resampled_nii: spatialimages.SpatialImage = processing.resample_from_to(
        source_nii, target_nii, **resample_kwargs
    )
    return resampled_nii


def resample_and_orient_nifti(
    source_nii: spatialimages.SpatialImage | GESeries.NiiBase,
    target_nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = DEFAULT_PARAMS["orientation"],
    display_plane: DISPLAY_PLANES | None = None,
    **kwargs: Any,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Resample `source_nii` onto `target_nii`'s grid, then orient both for display.

    Parameters
    ----------
    source_nii : spatialimages.SpatialImage or GESeries.NiiBase
        The image to resample.
    target_nii : spatialimages.SpatialImage or GESeries.NiiBase
        The image whose grid to resample onto.
    orientation : tuple[str, str, str], optional
        Target axis codes passed to `orient_nifti`. Defaults to
        `DEFAULT_PARAMS["orientation"]`.
    display_plane : {"axial", "coronal", "sagittal"}, optional
        If given, overrides `orientation` with a preset for the named plane.
    **kwargs
        Additional keyword arguments passed to `resample_nifti`.

    Returns
    -------
    tuple[ndarray, ndarray]
        The oriented target image data and the oriented resampled data.
    """
    if isinstance(source_nii, GESeries.NiiBase):
        source_nii: spatialimages.SpatialImage = source_nii.nii
    if isinstance(target_nii, GESeries.NiiBase):
        target_nii: spatialimages.SpatialImage = target_nii.nii
    # Resample base_nifti to target_nifti space
    resampled_nii = resample_nifti(source_nii=source_nii, target_nii=target_nii, **kwargs)
    # Orient the data
    base_images = orient_nifti(target_nii, orientation, display_plane=display_plane)
    data_images = orient_nifti(resampled_nii, orientation, display_plane=display_plane)
    return base_images, data_images
