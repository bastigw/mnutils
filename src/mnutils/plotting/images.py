from __future__ import annotations

import io
from typing import Any, TypedDict, Unpack

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from IPython.display import display as ipy_display
from loguru import logger
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from nibabel import affines, spatialimages

from .. import GESeries, utils
from ._widgets import MRSIVoxelInspectorWidget, SliceViewerWidget

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
        f"Sampling data with step size {sample_step} to estimate bounds "
        f"from a total of {data.size} elements."
    )
    # First filter out nan values and then sample the data to ensure we are
    # sampling valid values for bounds estimation
    sampled_data = data[np.isfinite(data)]
    sampled_data = sampled_data[::sample_step]

    if sampled_data.size == 0:
        logger.warning("Sampled data contains only NaN values. Returning (0, 0) as bounds.")
        return 0.0, 1.0

    # Calculate the lower and upper percentiles of the sampled data
    lower_bound, upper_bound = np.percentile(sampled_data, [p_lower, p_upper])
    logger.trace(
        f"Sampled {sampled_data.size} elements for bounds estimation "
        f"after excluding NaN values."
        f"\nEstimated bounds: lower={lower_bound:2.2e}, upper={upper_bound:2.2e} "
        f"using percentiles {p_lower} and {p_upper}."
    )

    return lower_bound, upper_bound


def display_nifti(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = None,
    **kwargs,
) -> tuple[Figure, npt.NDArray] | None:
    """Orient a NIfTI image and display it via `display_images`.

    Parameters
    ----------
    nii : nibabel.spatialimages.SpatialImage or GESeries.NiiBase
        The NIfTI image (or wrapper) to display.
    orientation : tuple[str, str, str], optional
        Target axis orientation codes, by default None.
    **kwargs
        Additional keyword arguments forwarded to `display_images`.

    Returns
    -------
    tuple[Figure, ndarray] or None
        See `display_images`.
    """
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
) -> tuple[Figure, npt.NDArray] | None:
    """Display a grid of 2D/3D/4D images, with a slice slider for 3D/4D input.

    Parameters
    ----------
    images : ndarray or list
        Image data (2D, 3D, or 4D) or a list of arrays to stack along a new
        last axis.
    titles : list[str], optional
        Titles for each image, by default None.
    fig_title : str, optional
        Overall figure title, by default "".
    cmap : str, optional
        Colormap to use, by default the module's default colormap.
    imshow_kws : dict, optional
        Extra keyword arguments passed to `imshow`.
    colorbar : bool, optional
        Whether to display a colorbar, by default False.
    colorbar_kws : dict, optional
        Extra keyword arguments for the colorbar; supports a `"mode"` key
        of `"single"` or `"each"`.
    fig_kws : dict, optional
        Extra keyword arguments passed to `plt.figure`.
    vmin : float, optional
        Lower display bound. Estimated from the data if omitted.
    vmax : float, optional
        Upper display bound. Estimated from the data if omitted.
    v_percentile : float, optional
        Percentile used to estimate `vmin`/`vmax` when not given, by
        default 1.0.
    zeros_as_nan : bool, optional
        If True, treat zero values as NaN for display, by default False.
    **kwargs
        Additional axis parameters merged into the default image axis
        parameters.

    Returns
    -------
    tuple[Figure, ndarray] or None
        For 2D input (or 3D/4D input with only one slice), the created
        figure and its axes. For 3D/4D input with more than one slice, an
        interactive slice-scrubbing widget is displayed instead and `None`
        is returned.
    """
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
                "All values in images are 0, so setting zeros_as_nan to False "
                "to avoid all values being NaN."
            )
            zeros_as_nan = False
            zeros_nan_images = images
        images = zeros_nan_images

    if vmin is None or vmax is None:
        new_vmin, new_vmax = fast_bounds(images, p_lower=v_percentile, p_upper=100 - v_percentile)
        if vmin is None:
            vmin = new_vmin
        if vmax is None:
            vmax = new_vmax

    if vmin == vmax or vmin is None or vmax is None or np.isnan([vmin, vmax]).any():
        logger.debug(
            f"vmin ({vmin}) and vmax ({vmax}) are equal or NaN. Adjusting vmin "
            f"and vmax to the min and max of the images for better visualization."
        )
        vmin, vmax = np.nanmin(images), np.nanmax(images)
        # If vmin and vmax are still equal after adjustment, set vmin to 0 to
        # avoid errors in imshow
        if vmin == vmax:
            logger.debug(
                "vmin and vmax are still equal after adjustment. This is likely "
                "to all non zeros values being equal (e.g. in a mask). Setting "
                "vmin to 0 for better visualization."
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

    grid_kwargs = dict(
        num_images=num_images,
        num_rows=num_rows,
        num_cols=num_cols,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        imshow_kws=imshow_kws,
        image_params=image_params,
        titles=titles,
        colorbar=colorbar,
        colorbar_kws=colorbar_kws,
        colorbar_mode=colorbar_mode,
        fig_title=fig_title,
    )

    n_slices = images.shape[2]
    if original_dims >= 3 and n_slices > 1:
        frames = _render_image_grid_frames(
            images, list(range(n_slices)), fig_size=fig_size, fig_kws=fig_kws, **grid_kwargs
        )
        ipy_display(SliceViewerWidget(frames=frames, n_slices=n_slices, initial_index=slice_idx))
        return None

    fig = plt.figure(figsize=fig_size, **fig_kws if fig_kws else {})
    axes = fig.subplots(num_rows, num_cols, sharex=True, sharey=True, squeeze=False).flatten()  # type: ignore
    _draw_image_grid(fig, axes, images, slice_idx, **grid_kwargs)
    return fig, axes


def _draw_image_grid(
    fig: Figure,
    axes: list[Axes],
    images: npt.NDArray,
    slice_idx: int,
    *,
    num_images: int,
    num_rows: int,
    num_cols: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    imshow_kws: dict[str, Any] | None,
    image_params: dict[str, Any],
    titles: list[str],
    colorbar: bool,
    colorbar_kws: dict[str, Any] | None,
    colorbar_mode: str,
    fig_title: str,
) -> tuple[list[AxesImage], list[Any]]:
    """Draw one slice of the `display_images` grid onto pre-built axes.

    Returns the per-axis image artists and colorbars (`None` where an axis
    has no colorbar), so a later slice can be applied in place via
    `_update_image_grid` without rebuilding the figure.
    """
    ims: list[AxesImage] = []
    cbars: list[Any] = []
    for image in range(num_images):
        im = axes[image].imshow(
            images[:, :, slice_idx, image],
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            **imshow_kws if imshow_kws else {},
        )
        ims.append(im)
        axes[image].set(**image_params)
        if len(titles) > image:
            im_row = image // num_cols
            if num_rows == 2 and im_row == 1:
                axes[image].set_xlabel(
                    titles[image], fontsize="medium", fontweight="normal", labelpad=6
                )
            else:
                axes[image].set_title(titles[image], fontsize="medium", fontweight="normal")

        cbar = None
        if colorbar:
            if colorbar_mode == "each":
                cbar = fig.colorbar(im, ax=axes[image], **colorbar_kws if colorbar_kws else {})
            elif colorbar_mode == "single" and image == num_images - 1:
                cbar = fig.colorbar(im, ax=axes[image], **colorbar_kws if colorbar_kws else {})
        cbars.append(cbar)

    for empty_ax in range(num_images, len(axes)):
        fig.delaxes(axes[empty_ax])

    if fig_title:
        fig.suptitle(fig_title, fontsize="large", fontweight="bold", y=1)

    return ims, cbars


def _update_image_grid(
    images: npt.NDArray,
    slice_idx: int,
    ims: list[AxesImage],
    cbars: list[Any],
    *,
    num_images: int,
    colorbar_mode: str,
    **_unused,
) -> None:
    """Apply a new slice to an already-drawn `_draw_image_grid` in place.

    Titles, axis params, and (for `colorbar_mode == "single"`) the colorbar
    are all slice-independent and were already set by `_draw_image_grid`, so
    only the image data (and, for `colorbar_mode == "each"`, the per-axis
    autoscale + colorbar range) needs updating per slice.
    """
    for image in range(num_images):
        ims[image].set_data(images[:, :, slice_idx, image])
        if colorbar_mode == "each":
            ims[image].autoscale()
            cbars[image].update_normal(ims[image])


def _render_image_grid_frames(
    images: npt.NDArray,
    slice_indices: list[int],
    *,
    fig_size: tuple[float, float],
    fig_kws: dict[str, Any] | None,
    **grid_kwargs,
) -> list[bytes]:
    """Render a batch of `display_images` slices to PNG bytes.

    Builds one Figure/axes for the whole batch (off the pyplot registry) and
    reuses it across slices via `_update_image_grid`, instead of rebuilding
    the figure/gridspec/colorbar from scratch per slice.
    """
    fig = Figure(figsize=fig_size, **fig_kws if fig_kws else {})
    FigureCanvasAgg(fig)
    num_rows, num_cols = grid_kwargs["num_rows"], grid_kwargs["num_cols"]
    axes = fig.subplots(num_rows, num_cols, sharex=True, sharey=True, squeeze=False).flatten()

    frames = []
    ims: list[AxesImage] | None = None
    cbars: list[Any] = []
    for slice_idx in slice_indices:
        if ims is None:
            ims, cbars = _draw_image_grid(fig, axes, images, slice_idx, **grid_kwargs)
        else:
            _update_image_grid(images, slice_idx, ims, cbars, **grid_kwargs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        frames.append(buf.getvalue())
    return frames


def overlay_nifti_data_on_T1(
    t1_nii: spatialimages.SpatialImage,
    data_nii: spatialimages.SpatialImage,
    orientation: tuple[str, str, str] | None = None,
    display_plane: utils.nifti.DISPLAY_PLANES | None = None,
    resample_kwargs: dict[str, Any] | None = None,
    mask: npt.NDArray | GESeries.NiiBase | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]] | None:
    """Resample a data NIfTI into T1 space and overlay it on the T1 image.

    Parameters
    ----------
    t1_nii : nibabel.spatialimages.SpatialImage
        Anatomical T1 NIfTI image to overlay onto.
    data_nii : nibabel.spatialimages.SpatialImage
        Data NIfTI image to resample and overlay.
    orientation : tuple[str, str, str], optional
        Target axis orientation codes, by default None.
    display_plane : utils.nifti.DISPLAY_PLANES, optional
        Anatomical plane to display, by default None.
    resample_kwargs : dict, optional
        Extra keyword arguments passed to `resample_and_orient_nifti`.
    mask : ndarray or GESeries.NiiBase, optional
        Boolean mask applied to the data before or after resampling,
        depending on whether its shape matches the data or the T1 image.
    **kwargs
        Additional keyword arguments forwarded to `overlay_image_data_on_T1`.

    Returns
    -------
    tuple[Figure, list[Axes]] or None
        See `overlay_image_data_on_T1`.
    """
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
                        "Mask shape matches T1 NIfTI but not data NIfTI. "
                        "Will apply mask after resampling."
                    )
                    apply_mask_after_resampling = True
                else:
                    raise ValueError(
                        f"Mask must have the same shape as data NIfTI.\n"
                        f"Shape of mask: {mask.shape}, "
                        f"shape of data NIfTI: {data_nii.shape}"
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

    return overlay_image_data_on_T1(t1_images, data_images, mask=mask_to_pass_on, **kwargs)


def overlay_image_data_on_T1(
    t1_images: npt.NDArray,
    data_images: npt.NDArray,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    only_overlay: bool = False,
    mask_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]] | None:
    """Display T1, resampled data, and overlay images side by side with a slice slider.

    Parameters
    ----------
    t1_images : ndarray
        T1-weighted anatomical image data.
    data_images : ndarray
        Data image, resampled into T1 space, to overlay.
    mask : ndarray, optional
        Boolean mask applied to the overlay data, by default None.
    mask_contour : ndarray or bool, optional
        Mask (or `True` to reuse `mask`) whose contour is drawn on the
        overlay axis, by default None.
    only_overlay : bool, optional
        If True, only show the overlay axis instead of T1, data, and
        overlay side by side, by default False.
    mask_kwargs : dict, optional
        Keyword arguments customizing the mask contour's appearance.
    **kwargs
        Additional keyword arguments forwarded to
        `overlay_image_data_on_T1_on_ax`.

    Returns
    -------
    tuple[Figure, list[Axes]] or None
        For a single-slice input, the created figure and its axes. For
        multi-slice input, an interactive slice-scrubbing widget is
        displayed instead and `None` is returned.
    """
    if data_images.shape != t1_images.shape:
        raise ValueError(
            f"T1 images and data images must have the same shape for overlay. "
            f"T1 shape: {t1_images.shape}, data shape: {data_images.shape}"
        )

    if t1_images.ndim < 3:
        slice_idx = 0
        t1_images = t1_images[:, :, np.newaxis]
        data_images = data_images[:, :, np.newaxis]
        if mask is not None:
            mask = mask[:, :, np.newaxis]
        if isinstance(mask_contour, np.ndarray):
            mask_contour = mask_contour[:, :, np.newaxis]
    else:
        slice_idx = t1_images.shape[2] // 2

    # `mask_contour` only draws a contour when it's `True` (reuse `mask`); an
    # explicit array is not applied here (matches the pre-anywidget behavior).
    resolved_mask_contour = mask if mask_contour is True else None
    contour_kwargs = DEFAULT_MASK_PARAMS.copy()
    if mask_kwargs is not None:
        contour_kwargs.update(mask_kwargs)

    overlay_kwargs = dict(
        only_overlay=only_overlay,
        mask=mask,
        resolved_mask_contour=resolved_mask_contour,
        mask_kwargs=mask_kwargs,
        kwargs=kwargs,
    )

    n_slices = t1_images.shape[2]
    if n_slices > 1:
        frames = _render_overlay_frames(
            t1_images, data_images, list(range(n_slices)), **overlay_kwargs
        )
        ipy_display(SliceViewerWidget(frames=frames, n_slices=n_slices, initial_index=slice_idx))
        return None

    fig = plt.figure(figsize=(12, 4))
    axes = [fig.subplots(1, 1)] if only_overlay else list(fig.subplots(1, 3))
    _draw_overlay_slice(fig, axes, t1_images, data_images, slice_idx, **overlay_kwargs)
    return fig, axes


def _draw_overlay_slice(
    fig: Figure,
    axes: list[Axes],
    t1_images: npt.NDArray,
    data_images: npt.NDArray,
    slice_idx: int,
    *,
    only_overlay: bool,
    mask: npt.NDArray[np.bool_] | None,
    resolved_mask_contour: npt.NDArray[np.bool_] | None,
    mask_kwargs: dict[str, Any] | None,
    kwargs: dict[str, Any],
) -> None:
    """Draw one slice of the `overlay_image_data_on_T1` panels onto given axes."""
    axes_idx = 0
    if not only_overlay:
        overlay_image_data_on_T1_on_ax(
            t1_images[:, :, slice_idx], ax=axes[axes_idx], cmap="gray", **kwargs
        )
        axes[axes_idx].set_title("T1 Image")
        axes_idx += 1
        overlay_image_data_on_T1_on_ax(data_images[:, :, slice_idx], ax=axes[axes_idx], **kwargs)
        axes[axes_idx].set_title("Resampled Data")
        axes_idx += 1

    overlay_image_data_on_T1_on_ax(
        t1_images[:, :, slice_idx],
        data_images[:, :, slice_idx],
        mask=mask[:, :, slice_idx] if mask is not None else None,
        mask_contour=resolved_mask_contour[:, :, slice_idx]
        if resolved_mask_contour is not None
        else None,
        ax=axes[axes_idx],
        mask_kwargs=mask_kwargs,
        **kwargs,
    )
    axes[axes_idx].set_title("Overlay")


def _render_overlay_frames(
    t1_images: npt.NDArray, data_images: npt.NDArray, slice_indices: list[int], **kwargs
) -> list[bytes]:
    """Render a batch of `overlay_image_data_on_T1` slices to PNG bytes.

    Builds one Figure/axes for the whole batch (off the pyplot registry) and
    reuses it across slices, clearing each axis before redrawing — the
    per-slice content (imshow bounds, mask contour) isn't reusable in place
    since it's recomputed per slice, but the figure/gridspec construction is.
    """
    only_overlay = kwargs["only_overlay"]
    fig = Figure(figsize=(12, 4))
    FigureCanvasAgg(fig)
    axes = [fig.subplots(1, 1)] if only_overlay else list(fig.subplots(1, 3))

    frames = []
    for i, slice_idx in enumerate(slice_indices):
        if i:
            for ax in axes:
                ax.cla()
        _draw_overlay_slice(fig, axes, t1_images, data_images, slice_idx, **kwargs)
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        frames.append(buf.getvalue())
    return frames


def overlay_image_data_on_T1_on_ax(
    base_image: npt.NDArray,
    overlay_image: npt.NDArray | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    ax: Axes | None = None,
    cmap: str = DEFAULT_PARAMS["cmap"],
    alpha: float = 0.5,
    mask_kwargs: dict[str, Any] | None = None,
    figsize: tuple[int, int] = (4, 4),
    **kwargs,
) -> (
    tuple[list[AxesImage], QuadContourSet | None]
    | tuple[Figure, Axes, list[AxesImage], QuadContourSet | None]
):
    """Draw a single 2D anatomical image, optionally with a data overlay/mask.

    If `ax` is omitted, creates its own figure and axes and returns them
    alongside the plotted images/contours.
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

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
    # If there is a base image and an overlay image, change the cmap for the
    # base image to gray and the overlay image to the provided cmap
    if overlay_image is not None:
        cmap_base = "gray"
    else:
        cmap_base = cmap

    # Get vmin and vmax bounds for anatomical image by default
    anat_vmin, anat_vmax = fast_bounds(base_image, p_lower=1.0, p_upper=99.0)

    im.append(ax.imshow(base_image, origin="upper", cmap=cmap_base, vmin=anat_vmin, vmax=anat_vmax))

    # See if vmin and vmax are provided in kwargs and apply to data image
    if overlay_image is None:
        ax.set(**image_params)
        if created_fig is not None:
            return created_fig, ax, im, None
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
        raise ValueError("T1 images and data images must have the same shape for overlay.")
    if mask is not None and mask.shape != overlay_image.shape:
        raise ValueError(
            f"Mask must have the same shape as data images. "
            f"Mask shape: {mask.shape}, data images shape: {overlay_image.shape}"
        )
    if mask is not None:
        overlay_image = np.where(mask, overlay_image, np.nan)
        logger.trace(
            f"Applied mask to overlay image. Mask shape: {mask.shape}, mask dtype: {mask.dtype}"
        )

    im.append(
        ax.imshow(
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
        contours = ax.contour(mask_contour, **contour_kwargs)
    else:
        contours = None

    ax.set(**image_params)
    if created_fig is not None:
        return created_fig, ax, im, contours
    return im, contours


def inspect_MRSI_spectra(
    T1: GESeries.MRISeries,
    MRSI: GESeries.MRSISeries,
    blocky: bool = True,
    magnitude: bool = False,
    autophase: bool = True,
) -> None:
    """Display an interactive widget to inspect MRSI spectra over a T1 image.

    Shows the T1 image with an MRSI overlay and a spectrum plot; clicking,
    the keyboard arrows, or the slice slider update the selected voxel and
    its spectrum. All interaction runs client-side (no Python kernel needed
    after the widget is displayed): every anatomical slice is pre-rendered,
    and every voxel's spectrum is precomputed into a single buffer embedded
    in the widget.

    Parameters
    ----------
    T1 : GESeries.MRISeries
        The T1-weighted anatomical series.
    MRSI : GESeries.MRSISeries
        The MRSI series to inspect.
    blocky : bool, optional
        If True, resample the raw (blocky) MRSI grid rather than the
        interpolated NIfTI, by default True.
    magnitude : bool, optional
        If True, display the magnitude spectrum, by default False.
    autophase : bool, optional
        If True (and `magnitude` is False), autophase the spectrum before
        display, by default True.
    """
    t1_images = T1.images()
    if blocky:
        t1_images, mrsi_images = utils.nifti.resample_and_orient_nifti(
            source_nii=MRSI.RAW_exp.nii,
            target_nii=T1.nii,
        )
    else:
        t1_images, mrsi_images = utils.nifti.resample_and_orient_nifti(
            source_nii=MRSI.nii, target_nii=T1.nii
        )

    n_anat_slices = t1_images.shape[2]
    initial_slice = n_anat_slices // 2

    blocky_mrsi_affine = np.asarray(MRSI.RAW_exp.nii.affine)
    t1_display_affine = utils.nifti.get_display_affine(T1.nii)
    mrsi_to_display = np.linalg.inv(t1_display_affine).dot(blocky_mrsi_affine)
    display_to_mrsi = np.linalg.inv(mrsi_to_display)

    logger.debug(f"T1 display affine:\n{t1_display_affine}")
    logger.debug(f"MRSI affine:\n{blocky_mrsi_affine}")
    logger.debug(f"MRSI to display affine:\n{mrsi_to_display}")

    nx, ny, n_mrsi_slices = MRSI.spec.shape[1], MRSI.spec.shape[2], MRSI.spec.shape[3]
    initial_voxel = (nx // 2, ny // 2, n_mrsi_slices // 2)

    anat_vmin, anat_vmax = fast_bounds(t1_images)
    mrsi_vmin, mrsi_vmax = fast_bounds(mrsi_images)
    logger.debug(f"Anatomical image bounds for display: vmin={anat_vmin:4g}, vmax={anat_vmax:4g}")
    logger.debug(f"MRSI image bounds for display: vmin={mrsi_vmin:4g}, vmax={mrsi_vmax:4g}")

    def mrsi_slice_for(anat_slice_idx: int) -> int:
        raw = affines.apply_affine(display_to_mrsi, [0, 0, anat_slice_idx])[2]
        return int(np.clip(np.round(raw), 0, n_mrsi_slices - 1))

    logger.debug("Starting rendering of Left frames")

    left_frames = _render_mrsi_left_frames(
        t1_images,
        mrsi_images,
        list(range(n_anat_slices)),
        anat_vmin=anat_vmin,
        anat_vmax=anat_vmax,
        mrsi_vmin=mrsi_vmin,
        mrsi_vmax=mrsi_vmax,
    )
    slice_titles = [
        f"Anat. Slice {s}, MRSI Slice {mrsi_slice_for(s)}" for s in range(n_anat_slices)
    ]

    if magnitude and autophase:
        logger.warning(
            "Both magnitude and autophase options selected. Autophase will "
            "be ignored when displaying magnitude spectrum."
        )

    logger.debug("Starting rendering of spectra array")

    npts = MRSI.ppm.values.size
    spectral_dim = MRSI.spec.dims[0]
    # `autophase()` finds one phase correction from the single voxel with the
    # global-max signal and applies it to the whole grid in one call, instead
    # of an independent (and far slower) optimization per voxel -- see
    # docs/diary/2026-08-14-anywidget-slice-viewer.md for the per-voxel
    # timing that motivated this.
    if magnitude:
        spectra_array = np.abs(MRSI.spec)
    elif autophase:
        phased = MRSI.spec.xmr.to_hz().xmr.autophase()
        spectral_dim = (set(phased.dims) - {"i", "j", "k"}).pop()
        spectra_array = phased.real
    else:
        spectra_array = MRSI.spec.real
    # Top left of the image is (0, 0); the MRSI grid's coordinate system
    # starts bottom-right, so the array is reversed along i/j and the (x, y)
    # voxel axes read from (j, i) rather than (i, j) -- matches the mapping
    # `overlay_voxel_on_T1`/the widget's client-side voxel picker expect.
    spectra_array = (
        spectra_array.isel(i=slice(None, None, -1), j=slice(None, None, -1))
        .transpose("j", "i", "k", spectral_dim)
        .values.astype(np.float32)
    )

    if magnitude:
        spectrum_label = "Magnitude Spectrum"
    elif autophase:
        spectrum_label = "Autophased Spectrum"
    else:
        spectrum_label = "Real Spectrum"

    logger.debug("Ready to display")

    ipy_display(
        MRSIVoxelInspectorWidget(
            left_frames=left_frames,
            slice_titles=slice_titles,
            n_anat_slices=n_anat_slices,
            initial_slice=initial_slice,
            image_width=t1_images.shape[1],
            image_height=t1_images.shape[0],
            mrsi_to_display_affine=mrsi_to_display.flatten().tolist(),
            display_to_mrsi_affine=display_to_mrsi.flatten().tolist(),
            grid_shape=(nx, ny, n_mrsi_slices),
            mrsi_dims=(int(MRSI.dims[0]), int(MRSI.dims[1])),
            initial_voxel=initial_voxel,
            ppm=MRSI.ppm.values.astype(float).tolist(),
            spectra_bytes=spectra_array.tobytes(),
            npts=npts,
            spectrum_label=spectrum_label,
        )
    )


def _render_mrsi_left_frames(
    t1_images: npt.NDArray,
    mrsi_images: npt.NDArray,
    slice_indices: list[int],
    *,
    anat_vmin: float,
    anat_vmax: float,
    mrsi_vmin: float,
    mrsi_vmax: float,
) -> list[bytes]:
    """Render a batch of T1+MRSI-overlay anatomical slices, one PNG per slice.

    The axes fills the entire figure with no ticks/labels/margin and uses
    `aspect="auto"`, so each saved PNG's content area maps linearly onto the
    image's (ncols, nrows) data extent -- required for the widget's
    client-side voxel-outline overlay to align with the image. Since vmin/
    vmax are fixed across all slices, one Figure/axes/imshow pair is built
    for the whole batch and each slice just updates the image data in place.
    """
    nrows, ncols = t1_images.shape[:2]
    fig = Figure(figsize=(4, 4 * nrows / ncols))
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    im_anat = ax.imshow(
        t1_images[:, :, slice_indices[0]],
        cmap="gray",
        origin="upper",
        vmin=anat_vmin,
        vmax=anat_vmax,
    )
    im_mrsi = ax.imshow(
        mrsi_images[:, :, slice_indices[0]],
        cmap="magma",
        alpha=0.5,
        origin="upper",
        vmin=mrsi_vmin,
        vmax=mrsi_vmax,
    )
    ax.set_aspect("auto")
    ax.axis("off")

    frames = []
    for i, slice_idx in enumerate(slice_indices):
        logger.debug(f"Rendering slice {slice_idx} ({i + 1}/{len(slice_indices)})")
        if i:
            im_anat.set_data(t1_images[:, :, slice_idx])
            im_mrsi.set_data(mrsi_images[:, :, slice_idx])
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        frames.append(buf.getvalue())
    return frames


class VoxelOverlayParams(TypedDict, total=False):
    """Keyword arguments accepted by `overlay_voxel_on_T1`."""

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
            raise ValueError("Length of voxel_kwargs list must match the number of voxels.")
        resolved_voxel_kwargs = [default_voxel_kwargs | single for single in voxel_kwargs]
    else:
        resolved_voxel_kwargs = [default_voxel_kwargs | (voxel_kwargs or {})] * len(voxel_coords)

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
    ax: Axes | None = None,
    figsize: tuple[int, int] = (6, 6),
    **kwargs: Unpack[VoxelOverlayParams],
) -> tuple[Figure, Axes] | Axes:
    """Overlay one or more voxels (and optionally MRSI data) on a T1 slice.

    Displays a slice of a T1-weighted MRI image, overlays voxel(s) at the
    specified coordinates, and optionally overlays MRSI data resampled into
    T1 space. If `ax` is omitted, creates its own figure and axes.

    Parameters
    ----------
    t1 : GESeries.MRISeries
        The T1-weighted MRI series containing the anatomical image data.
    MRSI : GESeries.MRSISeries
        The MRSI series containing the spectroscopic imaging data.
    voxel_coords : tuple[int, int, int] or ndarray or list[tuple[int, int, int]]
        The coordinates of the voxel(s) in MRSI space. If a list is given,
        all voxels must share the same z-coordinate.
    ax : matplotlib.axes.Axes, optional
        The axis on which to plot the image and overlays. If `None`, a new
        figure and axis are created.
    figsize : tuple[int, int], optional
        The size of the figure in inches, used only when `ax` is `None`.
        Defaults to (6, 6).
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
    tuple[matplotlib.figure.Figure, matplotlib.axes.Axes] or matplotlib.axes.Axes
        The created `(fig, ax)` if `ax` was `None`, otherwise the axis with
        the plotted image and overlays.

    Notes
    -----
    The MRSI data is resampled to T1 space using the provided affine
    transformations, then scaled to `[0, 1]` if `scale_overlay` is True. The
    voxel rectangle(s) are drawn at the specified coordinates, adjusted to
    align with the anatomical space.
    """
    created_fig = None
    if ax is None:
        created_fig, ax = plt.subplots(figsize=figsize)

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
        logger.trace(f"Sum of MRSI image slice: {np.nansum(mrsi_images[:, :, slice_idx])}")

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

    if created_fig is not None:
        return created_fig, ax
    return ax
