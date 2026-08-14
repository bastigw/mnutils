from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

import nibabel as nib
import numpy as np
import numpy.typing as npt
from loguru import logger
from nibabel import affines, processing, spatialimages

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


def get_display_affine(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] = ("L", "P", "S"),
) -> npt.NDArray[np.float64]:
    """Return the affine mapping display voxel indices (after orient_nifti) to world mm.

    orient_nifti applies apply_orientation then flipud(rot90(k=1)) which equals
    a transpose of axes 0 and 1. This function encodes both steps as an affine.
    Use inv(get_display_affine(t1)) @ mrsi_affine to get mrsi→display mapping.
    """
    if isinstance(nii, GESeries.NiiBase):
        nii: spatialimages.SpatialImage = nii.nii
    current_ornt = nib.orientations.io_orientation(nii.affine)
    target_ornt = nib.orientations.axcodes2ornt(orientation)
    transform = nib.orientations.ornt_transform(current_ornt, target_ornt)
    reoriented = nii.as_reoriented(transform)  # pyright: ignore[reportArgumentType]
    oriented_affine = np.asarray(reoriented.affine, dtype=float)
    # flipud(rot90(k=1, axes=(0,1))) = pure transpose of axes 0 and 1.
    # This swaps columns 0 and 1 of the affine.
    P = np.eye(4)
    P[0, 0] = P[1, 1] = 0.0
    P[0, 1] = P[1, 0] = 1.0
    return oriented_affine @ P


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
        logger.debug(
            f"Orienting NIfTI to {display_plane} plane. Using orientation {orientation}."
        )

    if orientation is None:
        logger.debug(
            "No orientation specified. Using default orientation: "
            f"{DEFAULT_PARAMS['orientation']}."
        )
        orientation = DEFAULT_PARAMS["orientation"]

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
    resampled_nii = resample_nifti(
        source_nii=source_nii, target_nii=target_nii, **kwargs
    )
    # Orient the data
    base_images = orient_nifti(target_nii, orientation, display_plane=display_plane)
    data_images = orient_nifti(resampled_nii, orientation, display_plane=display_plane)
    return base_images, data_images


def downsample_to_coverage(
    src_data: npt.NDArray | spatialimages.SpatialImage | GESeries.NiiBase,
    tgt_data: spatialimages.SpatialImage | GESeries.NiiBase | None = None,
    src_affine: npt.NDArray | None = None,
    tgt_affine: npt.NDArray | None = None,
    tgt_shape: tuple[int, int, int] | None = None,
    super_sampling: bool = True,
    sampling_order: int | list[int] | tuple[int, int, int] | None = None,
    **kwargs,
) -> npt.NDArray[np.float32]:
    """Compute, per target voxel, the fraction of sub-samples that fall inside the source image.

    For each target voxel, samples a supersampling grid of sub-voxel points
    (or a single center point if `super_sampling` is False), maps them into
    source voxel space via the affines, and averages how many land inside
    the source image bounds.

    Parameters
    ----------
    src_data : ndarray, spatialimages.SpatialImage, or GESeries.NiiBase
        The source image data or image. If a plain array, `src_affine` must
        be given.
    tgt_data : spatialimages.SpatialImage or GESeries.NiiBase, optional
        The target image, used to derive `tgt_affine`/`tgt_shape` if not
        given directly.
    src_affine : ndarray, optional
        The source image's affine. Required if `src_data` is a plain array.
    tgt_affine : ndarray, optional
        The target image's affine. Required if `tgt_data` is not given.
    tgt_shape : tuple[int, int, int], optional
        The target image's shape. Required if `tgt_data` is not given.
    super_sampling : bool, optional
        Whether to sample a sub-voxel grid per target voxel (more accurate)
        or a single center point. Defaults to True.
    sampling_order : int, list[int], or tuple[int, int, int], optional
        The supersampling grid size per axis. Defaults to a size derived
        from the target-to-source voxel scale.
    **kwargs
        Unused, accepted for forward compatibility.

    Returns
    -------
    ndarray
        The coverage fraction for each target voxel, shape `tgt_shape`.

    Raises
    ------
    ValueError
        If the source/target affines or target shape cannot be determined.
    """
    if isinstance(src_data, GESeries.NiiBase):
        src_img = src_data.images()
        src_affine = src_data.nii.affine
    elif isinstance(src_data, spatialimages.SpatialImage):
        src_img = src_data.get_fdata()
        src_affine = src_data.affine
    else:
        src_img = src_data

    if isinstance(tgt_data, GESeries.NiiBase):
        tgt_affine = tgt_data.nii.affine
        tgt_shape = tgt_data.images().shape
    elif isinstance(tgt_data, spatialimages.SpatialImage):
        tgt_affine = tgt_data.affine
        tgt_shape = tgt_data.get_fdata().shape

    if src_affine is None or tgt_affine is None or tgt_shape is None:
        raise ValueError(
            "Source and target affines and target shape must be provided. Either pass "
            "target data as Nifti/GESeries or provide affines and shape directly."
        )

    # Generally we do not need to do a half voxel shift as our offsets center the samples
    # within the voxels
    # Double check by overlaying on the anatomical again
    tgt2src_affine = np.linalg.inv(src_affine).dot(tgt_affine)

    logger.debug(f"Source shape: {src_img.shape}, Target shape: {tgt_shape}")
    logger.debug(f"Transformation from target to source affine:\n{tgt2src_affine}")

    # Supersampling

    if super_sampling:
        if sampling_order is not None:
            if isinstance(sampling_order, int):
                s = [sampling_order] * 3
            else:
                s = list(sampling_order)
        else:
            s = np.ceil(np.diag(tgt2src_affine)[:3]).astype(int).tolist()
        logger.debug(f"Using supersampling grid of size: {s}")

        offset = np.multiply(s, [0, 0, -1])
        logger.debug(f"Offset for sampling: {offset}")
        # I am not entirely sure why we need to offset z direction by the sampling coefficient...
        offsets = (np.indices(s).transpose(1, 2, 3, 0) + offset) / s
    else:
        logger.warning(
            "No supersampling. This may lead to inaccurate coverage estimates, especially "
            "for small voxels or large transformations."
        )
        offsets = np.array([[[[-0, -0, -0]]]])  # Does not seem to neet a offest..
        logger.debug(f"No supersampling. Using single sample at offset: {offsets}")

    # MAIN LOOP

    coverage = np.zeros(tgt_shape, dtype=np.float32)

    for i in range(tgt_shape[0]):
        for j in range(tgt_shape[1]):
            for k in range(tgt_shape[2]):
                # --- Compute index of sub-samples within target voxel ---
                # target index coordinate + offsets
                tgt_idx = np.array([i, j, k])[None, None, None, :] + offsets
                src_idx = affines.apply_affine(tgt2src_affine, tgt_idx)

                # Nearest neighbor sample
                xi = np.rint(src_idx[..., 0]).astype(int)
                yi = np.rint(src_idx[..., 1]).astype(int)
                zi = np.rint(src_idx[..., 2]).astype(int)

                # Boundary mask
                inside = (
                    (xi >= 0)
                    & (xi < src_img.shape[0])
                    & (yi >= 0)
                    & (yi < src_img.shape[1])
                    & (zi >= 0)
                    & (zi < src_img.shape[2])
                )

                samples = np.zeros_like(xi, dtype=np.float32)
                samples[inside] = src_img[xi[inside], yi[inside], zi[inside]].astype(
                    np.float32
                )

                # Average over samples → coverage fraction
                coverage[i, j, k] = samples.mean()

    return coverage.astype(np.float32)
