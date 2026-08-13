from __future__ import annotations

from typing import Any, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from ipywidgets import interact, widgets
from loguru import logger
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from nibabel import affines, spatialimages

from .. import GESeries, utils
from . import spectra

DEFAULT_IMAGE_AX_PARAMS = {
    "xlabel": "",
    "xticks": [],
    "ylabel": "",
    "yticks": [],
}

DEFAULT_PARAMS = {
    "cmap": "magma",
    "ticker_steps": [1, 2, 4],
}

DEFAULT_MASK_PARAMS = {
    "colors": "green",
    "linewidths": 3,
}


def fast_bounds(
    data: npt.NDArray,
    max_target_samples: int = 50000,
    p_lower: float = 1.0,
    p_upper: float = 99.0,
) -> tuple[float, float]:
    """Quickly estimate lower/upper display bounds by sampling a subset of the data.

    Samples a subset of the input data and calculates the specified
    percentiles to estimate bounds suitable for visualization.

    Parameters
    ----------
    data : ndarray
        The input data array from which to estimate the bounds.
    max_target_samples : int, optional
        The maximum number of samples to use for estimating bounds. Defaults
        to 50000.
    p_lower : float, optional
        The lower percentile to calculate. Defaults to 1.0 (the 1st
        percentile).
    p_upper : float, optional
        The upper percentile to calculate. Defaults to 99.0 (the 99th
        percentile).

    Returns
    -------
    tuple[float, float]
        The estimated lower and upper bounds for visualization.
    """
    # Sample the data by taking every nth element based on the sample_step
    sample_step = max(1, data.size // max_target_samples)
    logger.trace(
        f"Sampling data with step size {sample_step} to estimate bounds from a total of {data.size} elements."
    )
    # First filter out nan values and then sample the data to ensure we are sampling valid values for bounds estimation
    sampled_data = data[np.isfinite(data)]
    sampled_data = sampled_data[::sample_step]

    if sampled_data.size == 0:
        logger.warning(
            "Sampled data contains only NaN values. Returning (0, 0) as bounds."
        )
        return 0.0, 1.0

    # Calculate the lower and upper percentiles of the sampled data
    lower_bound, upper_bound = np.percentile(sampled_data, [p_lower, p_upper])
    logger.trace(
        f"Sampled {sampled_data.size} elements for bounds estimation after excluding NaN values."
        f"\nEstimated bounds: lower={lower_bound:2.2e}, upper={upper_bound:2.2e} using percentiles {p_lower} and {p_upper}."
    )

    return lower_bound, upper_bound


def display_nifti(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = None,
    **kwargs,
) -> tuple[Figure, npt.NDArray]:
    if isinstance(nii, GESeries.NiiBase):
        nii = nii.nii
    return display_images(utils.nifti.orient_nifti(nii, orientation), **kwargs)


def display_images(
    images: npt.NDArray | list,
    titles: list[str] | None = None,
    fig_title: str = "",
    cmap: str = DEFAULT_PARAMS["cmap"],
    imshow_kws: dict[str, Any] | None = None,
    colorbar: bool = False,
    colorbar_kws: dict[str, Any] | None = None,
    fig_kws: dict[str, Any] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    v_percentile: float = 1.0,
    zeros_as_nan: bool = False,
    **kwargs,
) -> tuple[Figure, npt.NDArray]:
    if titles is None:
        titles = []

    # Copy default image params and update with any provided kwargs
    # TODO add option to set vmin and vmax
    image_params = DEFAULT_IMAGE_AX_PARAMS.copy()
    image_params.update(kwargs)

    # Convert list of images to 4D numpy array
    if isinstance(images, list):
        images = np.stack(images, axis=-1)

    # Cast images to float32 for display
    images = images.astype(np.float32)
    original_dims = images.ndim

    if original_dims == 2:
        num_images = 1
        num_cols, num_rows = 1, 1
        fig_size = (4, 4)
        slice_idx = 0
        images = images[:, :, np.newaxis, np.newaxis]
    elif original_dims == 3:
        num_images = 1
        num_cols, num_rows = 1, 1
        fig_size = (4, 4)
        # Put the images into a 4D array for easier handling later
        slice_idx = images.shape[2] // 2
        images = images[:, :, :, np.newaxis]
    elif original_dims == 4:
        num_images = images.shape[3]
        # Add four images per row
        num_rows = (num_images + 3) // 4
        num_cols = min(4, num_images)
        fig_size = (num_cols * 3.5, num_rows * 3)
        slice_idx = images.shape[2] // 2
        logger.debug(f"Displaying {num_images} images in a {num_rows}x{num_cols} grid.")
        # Check that 3rd dimension is the same for all images
        for i in range(1, num_images):
            if images.shape[2] != images[:, :, :, i].shape[2]:
                raise ValueError(
                    "All images must have the same number of slices in the 3rd dimension."
                )
    else:
        raise ValueError("Input images must be either 3D or 4D.")

    # Set 0 values in images to nan for better visualization and colorbar scaling
    if vmin is None or vmax is None or zeros_as_nan:
        logger.debug(
            "Setting 0 values in images to NaN for better visualization and colorbar scaling."
        )
        zeros_nan_images = images.copy()
        zeros_nan_images[zeros_nan_images == 0] = np.nan
        # Check that now not all values are nan
        if np.isnan(zeros_nan_images).all():
            logger.warning(
                "All values in images are 0, so setting zeros_as_nan to False to avoid all values being NaN."
            )
            zeros_as_nan = False
            zeros_nan_images = images
        images = zeros_nan_images

    if vmin is None or vmax is None:
        new_vmin, new_vmax = fast_bounds(
            images, p_lower=v_percentile, p_upper=100 - v_percentile
        )
        if vmin is None:
            vmin = new_vmin
        if vmax is None:
            vmax = new_vmax

    if vmin == vmax or vmin is None or vmax is None or np.isnan([vmin, vmax]).any():
        logger.debug(
            f"vmin ({vmin}) and vmax ({vmax}) are equal or NaN. Adjusting vmin and vmax to the min and max of the images for better visualization."
        )
        vmin, vmax = np.nanmin(images), np.nanmax(images)
        # If vmin and vmax are still equal after adjustment, set vmin to 0 to avoid errors in imshow
        if vmin == vmax:
            logger.debug(
                "vmin and vmax are still equal after adjustment. This is likely to all non zeros values being equal (e.g. in a mask). Setting vmin to 0 for better visualization."
            )
            vmin = 0

    logger.debug(f"Setting vmin: {vmin:4g}, vmax: {vmax:4g}")

    # Adjust aspect ratio based on image shape
    aspect_ratio = images.shape[1] / images.shape[0]
    kwargs.setdefault("aspect", aspect_ratio)
    logger.debug(f"Setting image aspect ratio to {kwargs['aspect']}.")

    # Copy default image params and update with any provided kwargs
    image_params = DEFAULT_IMAGE_AX_PARAMS.copy()
    image_params.update(kwargs)

    if colorbar:
        if colorbar_kws is not None:
            colorbar_mode: str = colorbar_kws.pop("mode", "single")
        else:
            colorbar_mode = "single"

        if colorbar_mode == "each":
            # Set default axes padding wider if each axis has its own colorbar
            vmin = None
            vmax = None
            logger.debug("Removing vmin and vmax as each axis has its own colorbar.")
    else:
        colorbar_mode = ""

    fig = plt.figure(figsize=fig_size, **fig_kws if fig_kws else {})
    subplot_kws = {}
    gridspec_kws = {}

    axes: list[Axes] = fig.subplots(
        num_rows,
        num_cols,
        sharex=True,
        sharey=True,
        squeeze=False,
        subplot_kw=subplot_kws,
        gridspec_kw=gridspec_kws,
    ).flatten()  # type: ignore
    logger.debug(f"Created figure with size {fig_size} and {len(axes)} axes.")

    ims = []
    for image in range(num_images):
        im = axes[image].imshow(
            images[:, :, slice_idx, image],
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **imshow_kws if imshow_kws else {},
        )
        axes[image].set(**image_params)
        # Set title if provided
        if len(titles) > image:
            # Calculate row number
            im_row = image // num_cols

            if num_rows == 2 and im_row == 1:
                logger.debug("Setting title as xlabel for 2-row layout.")
                axes[image].set_xlabel(
                    titles[image], fontsize="medium", fontweight="normal", labelpad=6
                )
            else:
                axes[image].set_title(
                    titles[image], fontsize="medium", fontweight="normal"
                )

        if colorbar:
            if colorbar_mode == "each":
                logger.debug(
                    f"Adding colorbar to axis {image} with parameters: {colorbar_kws}"
                )
                fig.colorbar(im, ax=axes[image], **colorbar_kws if colorbar_kws else {})
            elif colorbar_mode == "single" and image == num_images - 1:
                logger.debug(
                    f"Adding single colorbar to last axis with parameters: {colorbar_kws}"
                )
                fig.colorbar(im, ax=axes[image], **colorbar_kws if colorbar_kws else {})

        ims.append(im)

    # Remove all unused axes
    for empty_ax in range(num_images, len(axes)):
        fig.delaxes(axes[empty_ax])

    if fig_title:
        fig.suptitle(fig_title, fontsize="large", fontweight="bold", y=1)

    if original_dims >= 3:

        def update_slice(slice_idx):
            for image in range(num_images):
                ims[image].set_data(images[:, :, slice_idx, image])
            fig.canvas.draw_idle()

        interact(update_slice, slice_idx=(0, images.shape[2] - 1))

    return fig, axes


def overlay_nifti_data_on_T1(
    t1_nii: spatialimages.SpatialImage,
    data_nii: spatialimages.SpatialImage,
    orientation: tuple[str, str, str] | None = None,
    display_plane: utils.nifti.DISPLAY_PLANES | None = None,
    resample_kwargs: dict[str, Any] | None = None,
    mask: npt.NDArray | GESeries.NiiBase | None = None,
    **kwargs,
):
    # If mask in kwargs, apply mask to data_nii before resampling and overlaying
    apply_mask_after_resampling: bool = False
    mask_to_pass_on: npt.NDArray[np.bool_] | None = None
    if mask is not None:
        # assert mask is a numpy bool array
        if isinstance(mask, GESeries.NiiBase):
            mask_to_pass_on = mask.images(
                orientation=orientation, display_plane=display_plane
            ).astype(bool)
        elif isinstance(mask, np.ndarray) and mask.dtype == bool:
            if mask.shape != data_nii.shape:
                # Check if mask shape matches t1_shape than
                if mask.shape == t1_nii.shape:
                    logger.debug(
                        "Mask shape matches T1 NIfTI but not data NIfTI. Will apply mask after resampling."
                    )
                    apply_mask_after_resampling = True
                else:
                    raise ValueError(
                        f"Mask must have the same shape as data NIfTI.\nShape of mask: {mask.shape}, shape of data NIfTI: {data_nii.shape}"
                    )
            else:
                masked_data = np.where(mask, data_nii.get_fdata(), np.nan)
                data_nii = spatialimages.SpatialImage(
                    dataobj=masked_data,  # pyright: ignore[reportArgumentType]
                    affine=data_nii.affine,
                    header=data_nii.header,
                )
        else:
            raise ValueError("Mask must be a numpy array of boolean type.")

    # Resample data nifti to T1 nifti space before overlaying
    t1_images, data_images = utils.nifti.resample_and_orient_nifti(
        source_nii=data_nii,
        target_nii=t1_nii,
        orientation=orientation,
        display_plane=display_plane,
        **(resample_kwargs or {}),
    )

    if apply_mask_after_resampling and mask is not None:
        mask_to_pass_on = mask

    return overlay_image_data_on_T1(
        t1_images, data_images, mask=mask_to_pass_on, **kwargs
    )


def overlay_image_data_on_T1(
    t1_images: npt.NDArray,
    data_images: npt.NDArray,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    only_overlay: bool = False,
    mask_kwargs: dict[str, Any] | None = None,
    **kwargs,
):
    fig = plt.figure(figsize=(12, 4))
    if only_overlay:
        single_axes: Axes = fig.subplots(1, 1)
        axes = [single_axes]
    else:
        axes: list[Axes] = fig.subplots(1, 3)

    if data_images.shape != t1_images.shape:
        raise ValueError(
            f"T1 images and data images must have the same shape for overlay. T1 shape: {t1_images.shape}, data shape: {data_images.shape}"
        )

    if t1_images.ndim < 3:
        slice_idx = 0
        t1_images = t1_images[:, :, np.newaxis]
        data_images = data_images[:, :, np.newaxis]
    else:
        slice_idx = t1_images.shape[2] // 2

    axes_idx = 0
    if not only_overlay:
        ims, _ = overlay_image_data_on_T1_on_ax(
            axes[axes_idx],
            t1_images[:, :, slice_idx],
            cmap="gray",
            **kwargs,
        )
        im1 = ims[0]
        axes[axes_idx].set_title("T1 Image")
        logger.debug(
            f"Displayed T1 image with shape {t1_images.shape} at slice index {slice_idx}."
        )

        axes_idx += 1
        ims, _ = overlay_image_data_on_T1_on_ax(
            axes[axes_idx],
            data_images[:, :, slice_idx],
            **kwargs,
        )
        im2 = ims[0]
        axes[axes_idx].set_title("Resampled Data")
        axes_idx += 1
        logger.debug(
            f"Displayed resampled data image with shape {data_images.shape} at slice index {slice_idx}."
        )

    global contour, contour_kwargs, mask_contour_global
    if mask_contour is True:
        mask_contour_global = mask
    else:
        mask_contour_global = None

    contour_kwargs = DEFAULT_MASK_PARAMS.copy()
    if mask_kwargs is not None:
        contour_kwargs.update(mask_kwargs)

    ims, contour = overlay_image_data_on_T1_on_ax(
        axes[axes_idx],
        t1_images[:, :, slice_idx],
        data_images[:, :, slice_idx],
        mask=mask[:, :, slice_idx] if mask is not None else None,
        mask_contour=mask_contour_global[:, :, slice_idx]
        if mask_contour_global is not None
        else None,
        mask_kwargs=mask_kwargs,
        **kwargs,
    )
    im3, im4 = ims[0], ims[1]
    axes[axes_idx].set_title("Overlay")

    def update_slice(slice_idx):
        global contour, contour_kwargs, mask_contour_global
        if not only_overlay:
            im1.set_data(t1_images[:, :, slice_idx])  # pyright: ignore[reportPossiblyUnboundVariable]
            im2.set_data(data_images[:, :, slice_idx])  # pyright: ignore[reportPossiblyUnboundVariable]
        im3.set_data(t1_images[:, :, slice_idx])
        # Check if we need to mask the data image for the overlay
        if mask is not None:
            overlay_image = np.where(
                mask[:, :, slice_idx], data_images[:, :, slice_idx], np.nan
            )
        else:
            overlay_image = data_images[:, :, slice_idx]
        im4.set_data(overlay_image)
        if contour is not None:
            if mask_contour_global is False:
                mask_contour_global = None
            elif mask_contour_global is True:
                mask_contour_global = mask
            contour.remove()
            if isinstance(mask_contour_global, np.ndarray):
                contour = axes[axes_idx].contour(
                    mask_contour_global[:, :, slice_idx], **contour_kwargs
                )
        fig.canvas.draw_idle()

    interact(update_slice, slice_idx=(0, t1_images.shape[2] - 1))
    return fig, axes


def overlay_image_data_on_T1_on_ax(
    axes: Axes,
    base_image: npt.NDArray,
    overlay_image: npt.NDArray | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    cmap: str = DEFAULT_PARAMS["cmap"],
    alpha: float = 0.5,
    mask_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[list[AxesImage], QuadContourSet | None]:
    # Copy default image params and update with any provided kwargs
    remove_keys = ["vmin", "vmax", "v_percentile"]
    # Adjust aspect ratio based on image shape
    aspect_ratio = base_image.shape[1] / base_image.shape[0]
    kwargs.setdefault("aspect", aspect_ratio)
    logger.trace(f"Setting image aspect ratio to {kwargs['aspect']:.2f}.")
    image_params = DEFAULT_IMAGE_AX_PARAMS.copy()
    image_params.update(kwargs)
    image_params = {k: v for k, v in image_params.items() if k not in remove_keys}

    im: list[AxesImage] = []
    if overlay_image is None and mask is not None:
        base_image = np.where(mask, base_image, np.nan)
    # If there is a base image and a overlay image change the cmap for the base image to gray and overlay image to the provided cmap
    if overlay_image is not None:
        cmap_base = "gray"
    else:
        cmap_base = cmap

    # Get vmin and vmax bounds for anatomical image by default
    anat_vmin, anat_vmax = fast_bounds(base_image, p_lower=1.0, p_upper=99.0)

    im.append(
        axes.imshow(
            base_image, origin="upper", cmap=cmap_base, vmin=anat_vmin, vmax=anat_vmax
        )
    )

    # See if vmin and vmax are provided in kwargs and apply to data image
    if overlay_image is None:
        axes.set(**image_params)
        return im, None

    v_percentile = kwargs.pop("v_percentile", 1.0)
    vmin = kwargs.pop("vmin", None)
    vmax = kwargs.pop("vmax", None)
    if vmin is None or vmax is None:
        new_vmin, new_vmax = fast_bounds(
            overlay_image, p_lower=v_percentile, p_upper=100 - v_percentile
        )
        if vmin is None:
            vmin = new_vmin
        if vmax is None:
            vmax = new_vmax

    logger.debug(f"Setting overlay vmin: {vmin:4g}, vmax: {vmax:4g}")
    # Assert that t1_images and data_images have the same shape
    if base_image.shape != overlay_image.shape:
        raise ValueError(
            "T1 images and data images must have the same shape for overlay."
        )
    if mask is not None and mask.shape != overlay_image.shape:
        raise ValueError(
            f"Mask must have the same shape as data images. Mask shape: {mask.shape}, data images shape: {overlay_image.shape}"
        )
    if mask is not None:
        overlay_image = np.where(mask, overlay_image, np.nan)
        logger.trace(
            f"Applied mask to overlay image. Mask shape: {mask.shape}, mask dtype: {mask.dtype}"
        )

    im.append(
        axes.imshow(
            overlay_image,
            origin="upper",
            cmap=cmap,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )
    )

    if mask_contour is False:
        mask_contour = None
    elif mask_contour is True:
        mask_contour = mask

    if mask_contour is not None:
        mask_contour = mask_contour.astype(np.bool_)
        contour_kwargs = DEFAULT_MASK_PARAMS.copy()
        if mask_kwargs is not None:
            contour_kwargs.update(mask_kwargs)
        logger.debug(f"Applying mask contour with parameters: {contour_kwargs}")
        contours = axes.contour(mask_contour, **contour_kwargs)
    else:
        contours = None

    axes.set(**image_params)
    return im, contours


def inspect_MRSI_spectra(
    T1: GESeries.MRISeries,
    MRSI: GESeries.MRSISeries,
    blocky: bool = True,
    magnitude: bool = False,
    autophase: bool = True,
):
    with plt.ioff():
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    # Correctly orient the T1 and MRSI images and select the slice.
    # These are captured by the nested callbacks below via ``nonlocal`` instead
    # of leaking into module globals.
    markers: list = []
    t1_images = T1.images()
    if blocky:
        t1_images, MRSI_images = utils.nifti.resample_and_orient_nifti(
            source_nii=MRSI.RAW_exp.nii,
            target_nii=T1.nii,
        )
    else:
        t1_images, MRSI_images = utils.nifti.resample_and_orient_nifti(
            source_nii=MRSI.nii, target_nii=T1.nii
        )

    slice_idx = t1_images.shape[2] // 2

    # Ok now lets figure out how to go from x and y in the image to the voxel in the MRSI data
    blocky_mrsi_affine = np.asarray(MRSI.RAW_exp.nii.affine)
    t1_display_affine = utils.nifti.get_display_affine(T1.nii)
    mrsi_to_display = np.linalg.inv(t1_display_affine).dot(blocky_mrsi_affine)
    display_to_mrsi = np.linalg.inv(mrsi_to_display)

    logger.debug(f"T1 display affine:\n{t1_display_affine}")
    logger.debug(f"MRSI affine:\n{blocky_mrsi_affine}")
    logger.debug(f"MRSI to display affine:\n{mrsi_to_display}")

    MRSI_x, MRSI_y, MRSI_slice_idx = (
        MRSI.spec.shape[1] // 2,
        MRSI.spec.shape[2] // 2,
        MRSI.spec.shape[3] // 2,
    )
    mrsi_voxel_location: VoxelCoord = tuple(
        np.asarray([MRSI_x, MRSI_y, MRSI_slice_idx]).astype(int)
    )

    anat_vmin, anat_vmax = fast_bounds(t1_images)
    mrsi_vmin, mrsi_vmax = fast_bounds(MRSI_images)
    logger.debug(
        f"Anatomical image bounds for display: vmin={anat_vmin:4g}, vmax={anat_vmax:4g}"
    )
    logger.debug(
        f"MRSI image bounds for display: vmin={mrsi_vmin:4g}, vmax={mrsi_vmax:4g}"
    )

    aspect_ratio = t1_images.shape[1] / t1_images.shape[0]

    # Plot the corrected images
    im_bg = axs[0].imshow(
        t1_images[:, :, slice_idx],
        cmap="gray",
        origin="upper",
        vmin=anat_vmin,
        vmax=anat_vmax,
        aspect=aspect_ratio,
    )
    im_fg = axs[0].imshow(
        MRSI_images[:, :, slice_idx],
        cmap="magma",
        alpha=0.5,
        origin="upper",
        vmin=mrsi_vmin,
        vmax=mrsi_vmax,
        # aspect=aspect_ratio,
    )
    axs[0].set_title(f"Anat. Slice {slice_idx}, MRSI Slice {MRSI_slice_idx}")

    # Create graph with empty data to be updated later
    # Change label based on options
    if magnitude:
        label = "Magnitude Spectrum"
        if autophase:
            logger.warning(
                "Both magnitude and autophase options selected. Autophase will be ignored when displaying magnitude spectrum."
            )
    elif autophase:
        label = "Autophased Spectrum"
    else:
        label = "Real Spectrum"

    spectra_line = spectra.plot_spectra_on_ax(
        axs[1], MRSI.ppm.values, np.ones_like(MRSI.ppm), labels=[label]
    )

    def update_voxel_marker(mrsi_voxel_position: VoxelCoord):
        nonlocal markers
        for m in markers:
            m.remove()
        markers = draw_voxel_overlays_on_ax(
            ax=axs[0],
            voxel_coords=mrsi_voxel_position,
            mrsi_to_display_affine=mrsi_to_display,
        )

    def update(change):
        nonlocal slice_idx, MRSI_slice_idx
        slice_idx = change["new"]
        im_bg.set_data(t1_images[:, :, slice_idx])
        im_fg.set_data(MRSI_images[:, :, slice_idx])

        MRSI_slice_idx = np.round(
            affines.apply_affine(display_to_mrsi, [0, 0, slice_idx])
        ).astype(int)[2]
        axs[0].set_title(f"Anat. Slice {slice_idx}, MRSI Slice {MRSI_slice_idx}")
        MRSI_slice_idx = np.clip(MRSI_slice_idx, 0, MRSI.spec.shape[3] - 1)
        mrsi_voxel_location: VoxelCoord = tuple(
            np.asarray([MRSI_x, MRSI_y, MRSI_slice_idx]).astype(int)
        )
        update_voxel_marker(mrsi_voxel_location)
        update_spectra(mrsi_voxel_location)
        fig.canvas.draw_idle()

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            nonlocal slice_idx, MRSI_x, MRSI_y, MRSI_slice_idx
            # event.ydata = display row, event.xdata = display col
            mrsi_voxel_location: VoxelCoord = tuple(
                np.clip(
                    np.round(
                        affines.apply_affine(
                            display_to_mrsi, [event.ydata, event.xdata, slice_idx]
                        )
                    ),
                    [0, 0, 0],
                    [np.asarray(MRSI.dims) - 1],
                )
                .squeeze()
                .astype(int)
            )
            MRSI_x, MRSI_y, MRSI_slice_idx = mrsi_voxel_location
            update_voxel_marker(mrsi_voxel_location)
            update_spectra(mrsi_voxel_location)
            fig.canvas.draw_idle()

    def update_spectra(location: VoxelCoord):
        # Show magnitude or (autophased) spectrum
        x, y, MRSI_slice_idx = location
        # Top left of image is 0,0. The coordinate system otherwise starts bottom right
        # To select the correct voxel, we need to flip x and y coordinate
        # i in this case is
        spec_i = MRSI.dims[0] - 1 - int(y)
        spec_j = MRSI.dims[1] - 1 - int(x)

        complex_spectra = MRSI.get_voxel_spectrum(spec_i, spec_j, MRSI_slice_idx)
        mag_spec = np.abs(complex_spectra)
        if magnitude:
            spectra_data = mag_spec
        else:
            if autophase:
                spectra_data = complex_spectra.xmr.to_hz().xmr.autophase().real
            else:
                spectra_data = complex_spectra.real
        axs[1].set_title(
            f"Spectrum at voxel (i:{spec_i}, j:{spec_j}, slice:{MRSI_slice_idx}, sum:{np.sum(mag_spec):.5g})"
        )
        spectra_line[0].set_data(MRSI.ppm, spectra_data)
        axs[1].relim()
        axs[1].autoscale(enable=True, axis="y")

    def on_key(event):
        if event.key in ["up", "down", "left", "right"]:
            nonlocal MRSI_x, MRSI_y, MRSI_slice_idx
            if event.key == "down":
                MRSI_y = max(0, MRSI_y - 1)
            elif event.key == "up":
                MRSI_y = min(MRSI.spec.shape[2] - 1, MRSI_y + 1)
            elif event.key == "right":
                MRSI_x = max(0, MRSI_x - 1)
            elif event.key == "left":
                MRSI_x = min(MRSI.spec.shape[1] - 1, MRSI_x + 1)
            voxel_location: VoxelCoord = tuple(
                np.asarray([MRSI_x, MRSI_y, MRSI_slice_idx]).astype(int)
            )
            update_spectra(voxel_location)
            update_voxel_marker(voxel_location)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", onclick)

    update_voxel_marker(mrsi_voxel_location)
    update_spectra(mrsi_voxel_location)

    slider = widgets.IntSlider(value=slice_idx, min=0, max=t1_images.shape[2] - 1)
    slider.observe(update, names="value")
    return widgets.VBox([slider, fig.canvas])


class VoxelOverlayParams(TypedDict, total=False):
    show_overlay: bool
    voxel_kwargs: dict | list[dict] | None
    image_kwargs: dict | None
    overlay_kwargs: dict | None
    title: str | None
    scale_overlay: bool
    blocky: bool


VoxelCoord = tuple[int, int, int]
VoxelCoordsInput = VoxelCoord | npt.NDArray | list[VoxelCoord]
VoxelStyle = dict[str, Any]
VoxelStylesInput = VoxelStyle | list[VoxelStyle] | None


def get_voxel_visible_slice_range(
    voxel_coord: VoxelCoord,
    mrsi_to_display_affine: npt.NDArray,
    n_slices: int | None = None,
) -> tuple[int, int]:
    """Return inclusive third-dimension slice range where a voxel is visible."""
    # Voxel occupies [voxel-0.5, voxel+0.5] in each MRSI index dim (center convention).
    origin = np.asarray(voxel_coord, dtype=float) - 0.5
    corner_offsets = np.indices((2, 2, 2)).reshape(3, -1).T.astype(float)
    corners = origin + corner_offsets
    display_corners = affines.apply_affine(mrsi_to_display_affine, corners)
    min_slice = int(np.ceil(np.min(display_corners[:, 2])))
    max_slice = int(np.floor(np.max(display_corners[:, 2])))

    if n_slices is not None:
        min_slice = max(0, min_slice)
        max_slice = min(n_slices - 1, max_slice)

    return min_slice, max_slice


def draw_voxel_overlays_on_ax(
    ax: Axes,
    voxel_coords: VoxelCoord | list[VoxelCoord],
    mrsi_to_display_affine: npt.NDArray,
    voxel_kwargs: VoxelStylesInput = None,
) -> list[patches.Polygon]:
    """Draw voxel outlines on an axis.

    mrsi_to_display_affine must map MRSI voxel indices to display pixel coordinates
    (row, col) as produced by orient_nifti. Use utils.nifti.get_display_affine to
    obtain the display affine for the T1, then compute
    mrsi_to_display = inv(display_affine) @ MRSI.RAW_exp.nii.affine.

    The voxel's four in-plane corners (voxel index +/- 0.5 along axes 0 and 1) are
    projected through the full affine, so this is correct for display affines that
    transpose or rotate the axes -- reading the affine diagonal is not, because
    get_display_affine bakes an axis transpose into an anti-diagonal affine.
    """
    default_voxel_kwargs: VoxelStyle = {
        "linewidth": 2,
        "edgecolor": "cyan",
        "facecolor": "none",
    }

    if not isinstance(voxel_coords, list):
        voxel_coords = [voxel_coords]

    if isinstance(voxel_kwargs, list):
        if len(voxel_kwargs) != len(voxel_coords):
            raise ValueError(
                "Length of voxel_kwargs list must match the number of voxels."
            )
        resolved_voxel_kwargs = [
            default_voxel_kwargs | single for single in voxel_kwargs
        ]
    else:
        resolved_voxel_kwargs = [default_voxel_kwargs | (voxel_kwargs or {})] * len(
            voxel_coords
        )

    # In-plane corner offsets around the voxel centre (index +/- 0.5 on axes 0, 1).
    corner_offsets = np.array(
        [[-0.5, -0.5, 0.0], [0.5, -0.5, 0.0], [0.5, 0.5, 0.0], [-0.5, 0.5, 0.0]]
    )

    polys: list[patches.Polygon] = []
    for voxel, single_voxel_kwargs in zip(voxel_coords, resolved_voxel_kwargs):
        corners = np.asarray(voxel, dtype=float) + corner_offsets
        corners_in_display = affines.apply_affine(mrsi_to_display_affine, corners)
        # apply_affine yields (row, col, slice); Polygon wants (x, y) = (col, row).
        xy = corners_in_display[:, [1, 0]]
        logger.debug(f"Voxel {voxel} display corners (col, row):\n{xy}")

        range_start, range_end = get_voxel_visible_slice_range(
            voxel_coord=voxel,
            mrsi_to_display_affine=mrsi_to_display_affine,
        )
        logger.debug(
            f"Voxel {voxel} visible on display slices {range_start} to {range_end} (inclusive)."
        )

        poly = patches.Polygon(xy, closed=True, **single_voxel_kwargs)
        ax.add_patch(poly)
        polys.append(poly)

    return polys


def overlay_voxel_on_T1(
    t1: GESeries.MRISeries,
    MRSI: GESeries.MRSISeries,
    voxel_coords: VoxelCoordsInput,
    figsize: tuple[int, int] = (6, 6),
    **kwargs: Unpack[VoxelOverlayParams],
):
    """Overlay one or more voxels on a T1-weighted MRI image, in a new figure.

    Creates a matplotlib figure and axis, and overlays the specified voxel
    coordinates on the given T1-weighted image via `overlay_voxel_on_T1_on_ax`.

    Parameters
    ----------
    t1 : GESeries.MRISeries
        The T1-weighted MRI series to use as the background image.
    MRSI : GESeries.MRSISeries
        The MRSI series containing the voxel data.
    voxel_coords : tuple[int, int, int] or ndarray or list[tuple[int, int, int]]
        The coordinates of the voxel(s) to overlay. If a list is given, all
        voxels must share the same z-coordinate.
    figsize : tuple[int, int], optional
        The size of the figure in inches. Defaults to (6, 6).
    **kwargs
        Additional keyword arguments passed to `overlay_voxel_on_T1_on_ax`.

    Returns
    -------
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
        The created figure and axis with the overlay.
    """
    fig, ax = plt.subplots(figsize=figsize)
    overlay_voxel_on_T1_on_ax(
        ax=ax,
        t1=t1,
        MRSI=MRSI,
        voxel_coords=voxel_coords,
        **kwargs,
    )
    return fig, ax


def overlay_voxel_on_T1_on_ax(
    ax: Axes,
    t1: GESeries.MRISeries,
    MRSI: GESeries.MRSISeries,
    voxel_coords: VoxelCoordsInput,
    **kwargs: Unpack[VoxelOverlayParams],
):
    """Overlay one or more voxels (and optionally MRSI data) on a T1 slice.

    Displays a slice of a T1-weighted MRI image on the given axis, overlays
    voxel(s) at the specified coordinates, and optionally overlays MRSI data
    resampled into T1 space.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis on which to plot the image and overlays.
    t1 : GESeries.MRISeries
        The T1-weighted MRI series containing the anatomical image data.
    MRSI : GESeries.MRSISeries
        The MRSI series containing the spectroscopic imaging data.
    voxel_coords : tuple[int, int, int] or ndarray or list[tuple[int, int, int]]
        The coordinates of the voxel(s) in MRSI space. If a list is given,
        all voxels must share the same z-coordinate.
    **kwargs : Unpack[VoxelOverlayParams]
        show_overlay : bool, optional
            Whether to overlay the MRSI data on the T1 image. Defaults to True.
        scale_overlay : bool, optional
            Whether to scale the MRSI data to `[0, 1]` for better
            visualization when overlaid. Defaults to True.
        title : str, optional
            Title for the plot. Defaults to one naming the anatomical slice
            index.
        voxel_kwargs : dict or list[dict], optional
            Keyword arguments customizing the voxel rectangle(s)'
            appearance. If a list, it must match the number of voxels.
            Defaults to None.
        image_kwargs : dict, optional
            Keyword arguments customizing the T1 image's appearance.
        overlay_kwargs : dict, optional
            Keyword arguments customizing the MRSI overlay's appearance.

    Returns
    -------
    matplotlib.axes.Axes
        The axis with the plotted image and overlays.

    Notes
    -----
    The MRSI data is resampled to T1 space using the provided affine
    transformations, then scaled to `[0, 1]` if `scale_overlay` is True. The
    voxel rectangle(s) are drawn at the specified coordinates, adjusted to
    align with the anatomical space.
    """
    # Get elements from kwargs with defaults
    show_overlay: bool = kwargs.get("show_overlay", True)
    voxel_kwargs: VoxelStylesInput = kwargs.get("voxel_kwargs", None)
    image_kwargs: dict | None = kwargs.get("image_kwargs", None)
    overlay_kwargs: dict | None = kwargs.get("overlay_kwargs", None)
    blocky: bool = kwargs.get("blocky", True)

    if show_overlay:
        # Resample MRSI to T1 space
        t1_images, mrsi_images = utils.nifti.resample_and_orient_nifti(
            source_nii=MRSI.RAW_exp.nii if blocky else MRSI.nii,
            target_nii=t1.nii,
        )
        # Scale the mrsi images from 0 to 1 for better overlay
        if kwargs.get("scale_overlay", True):
            mrsi_images = (mrsi_images - np.nanmin(mrsi_images)) / (
                np.nanmax(mrsi_images) - np.nanmin(mrsi_images)
            )
    else:
        # Just get the T1 images
        t1_images = t1.images()
        mrsi_images = np.zeros_like(t1_images)

    logger.trace(f"T1 images shape: {t1_images.shape}")
    logger.trace(f"MRSI images shape: {mrsi_images.shape}")

    mrsi_affine = np.asarray(MRSI.RAW_exp.nii.affine)
    t1_display_affine = utils.nifti.get_display_affine(t1.nii)
    mrsi_to_display = np.linalg.inv(t1_display_affine).dot(mrsi_affine)

    logger.debug(f"T1 display affine:\n{t1_display_affine}")
    logger.debug(f"MRSI affine:\n{mrsi_affine}")
    logger.debug(f"MRSI to display affine:\n{mrsi_to_display}")

    # Ensure voxel_coords is a list of tuples
    if isinstance(voxel_coords, (tuple, np.ndarray)):
        voxel_coords = [tuple(voxel_coords)]
    elif not isinstance(voxel_coords, list):
        raise ValueError("voxel_coords must be a tuple, ndarray, or list of tuples.")

    # Check that all voxels have the same z-coordinate
    z_coords = {coord[2] for coord in voxel_coords}
    if len(z_coords) > 1:
        raise ValueError("All voxels must have the same z-coordinate.")

    slice_idx = int(np.round(affines.apply_affine(mrsi_to_display, voxel_coords[0])[2]))

    logger.debug(f"Displaying slice index: {slice_idx}")

    default_image_kwargs: dict[str, Any] = {"cmap": "gray"}
    default_overlay_kwargs: dict[str, Any] = {"cmap": "magma", "alpha": 0.5}

    # Update default kwargs with provided kwargs
    img_vmin, img_vmax = fast_bounds(t1_images[:, :, slice_idx])
    default_image_kwargs.update({"vmin": img_vmin, "vmax": img_vmax})
    image_kwargs = default_image_kwargs | (image_kwargs or {})

    ax.imshow(t1_images[:, :, slice_idx], origin="upper", **image_kwargs)
    logger.trace(f"Sum of t1 image slice: {np.nansum(t1_images[:, :, slice_idx])}")
    if show_overlay:
        mrsi_vmin, mrsi_vmax = fast_bounds(mrsi_images[:, :, slice_idx])
        overlay_kwargs = default_overlay_kwargs | (overlay_kwargs or {})
        overlay_kwargs.update({"vmin": mrsi_vmin, "vmax": mrsi_vmax})
        ax.imshow(
            mrsi_images[:, :, slice_idx],
            origin="upper",
            **overlay_kwargs,
        )
        logger.trace(
            f"Sum of MRSI image slice: {np.nansum(mrsi_images[:, :, slice_idx])}"
        )

    draw_voxel_overlays_on_ax(
        ax=ax,
        voxel_coords=voxel_coords,
        mrsi_to_display_affine=mrsi_to_display,
        voxel_kwargs=voxel_kwargs,
    )

    title: str | None = kwargs.get("title", None)
    if title != "":
        ax.set_title(title or f"Anatomical Slice {slice_idx}")

    ax.axis("off")
    return ax
