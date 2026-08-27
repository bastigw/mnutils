from __future__ import annotations

import io
import zlib
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, NamedTuple, TypedDict, Unpack, overload

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import PIL.Image
from IPython.display import display as ipy_display
from loguru import logger
from matplotlib import colormaps, patches
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.contour import QuadContourSet
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from nibabel import affines, spatialimages

from .. import GESeries, utils
from ..rcparams import rcParams, resolve_rc, validate_css_length
from ._widgets import ImageGridWidget, MRSIVoxelInspectorWidget, SliceViewerWidget

# WebP quality for the MRSI inspector's anatomical frames. These are display
# rasters -- already normalised, colormapped and alpha-blended to 8-bit -- not
# data, so lossy encoding costs nothing measurable: q92 holds ~41 dB PSNR
# against the exact composite while being ~7x smaller than the PNG equivalent.
MRSI_FRAME_QUALITY = 92

# WebP quality for `display_images` panels. Same reasoning as above -- these
# are colormapped 8-bit display rasters, not data -- but with an alpha channel,
# since NaN pixels stay transparent so the widget's background shows through.
GRID_FRAME_QUALITY = 92

# Below this many pixels a panel is encoded losslessly instead. Lossy WebP is
# tuned for photographs: on a 20x20 mask, where every pixel is a hard-edged
# block a reader looks *at* rather than through, and which the widget then
# blows up several times over, its ringing is plainly visible -- while the
# frame is a couple of kB either way. The threshold stays low because lossless
# is not free above it: at 90x90 it costs 4.4x the payload (0.85 -> 3.8 MB for
# a 700-panel grid) to fix artifacts nobody can see at that scale.
GRID_LOSSLESS_MAX_PIXELS = 64 * 64

# float16 tops out at 65504, and raw spectra routinely exceed that, so the
# buffer is divided by a scale factor before the cast and multiplied back in
# the browser. 32000 leaves an octave of headroom under the limit.
_F16_TARGET_MAX = 32000.0

# Warn past this uncompressed spectra size -- a big grid can otherwise silently
# produce a notebook hundreds of MB wide.
_SPECTRA_WARN_BYTES = 32 * 1024**2


def fast_bounds(
    data: npt.NDArray,
    max_target_samples: int = 50000,
    p_lower: float = 1.0,
    p_upper: float = 99.0,
    exclude_zeros: bool = False,
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
    exclude_zeros : bool, optional
        If True, drop exact zeros before taking the percentiles. Background
        voxels are usually exactly 0 and would otherwise dominate the lower
        percentile and flatten the contrast of the anatomy itself. Ignored
        when the data is all zeros -- a warning is logged and the zeros are
        kept so the bounds stay finite. Defaults to False.

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
    if exclude_zeros:
        nonzero_data = sampled_data[sampled_data != 0]
        if nonzero_data.size == 0 and sampled_data.size > 0:
            logger.warning(
                "All finite values are 0, so keeping the zeros for bounds estimation "
                "to avoid discarding the whole array."
            )
        elif nonzero_data.size > 0:
            sampled_data = nonzero_data
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


def _resolve_display_bounds(
    data: npt.NDArray,
    vmin: float | None,
    vmax: float | None,
    v_percentile: float = 1.0,
    exclude_zeros: bool = False,
) -> tuple[float, float]:
    """Fill in whichever of `vmin`/`vmax` was not given from the data itself.

    A caller is allowed to pin one bound and leave the other to the data, so
    the estimate has to run whenever *either* is missing and then replace only
    the missing one. `exclude_zeros` is handed straight to `fast_bounds`, so
    background zeros can be kept out of the estimate without also being masked
    out of what is drawn.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax
    new_vmin, new_vmax = fast_bounds(
        data,
        p_lower=v_percentile,
        p_upper=100 - v_percentile,
        exclude_zeros=exclude_zeros,
    )
    return (new_vmin if vmin is None else vmin, new_vmax if vmax is None else vmax)


def _widen_if_flat(low: float, high: float) -> tuple[float, float]:
    """Give a zero-width range one unit of span so `Normalize` has one to map.

    A constant panel (or an all-zero one under `zeros_as_nan`) still renders as
    the colormap's low end either way, but a colorbar reading "0 to 0" is worse
    than one reading "0 to 1".
    """
    return (low, high) if high > low else (low, low + 1.0)


def _encode_webp(
    array: npt.NDArray,
    mode: str | None = None,
    *,
    quality: int,
    lossless: bool = False,
) -> bytes:
    """Encode one display raster to WebP bytes.

    `method` trades encode time for size; the lossless path can afford the
    slower 4 because it is only ever taken for panels under
    `GRID_LOSSLESS_MAX_PIXELS`.
    """
    buf = io.BytesIO()
    image = PIL.Image.fromarray(array, mode)
    if lossless:
        image.save(buf, format="webp", lossless=True, method=4)
    else:
        image.save(buf, format="webp", quality=quality, method=2)
    return buf.getvalue()


def _mrsi_to_display_affine(
    t1_nii: spatialimages.SpatialImage, MRSI: GESeries.MRSISeries
) -> npt.NDArray:
    """Build the affine mapping raw MRSI voxel indices to T1 display pixels.

    The raw (blocky) grid is the reference in both directions: `RAW_exp`'s
    affine is what the voxel picker and the voxel outlines are drawn in, and
    `get_display_affine` bakes the orientation transpose that `orient_nifti`
    applies to the anatomical volume, so the inverse of the latter composed
    with the former is what carries one into the other.
    """
    mrsi_affine = np.asarray(MRSI.RAW_exp.nii.affine)
    t1_display_affine = utils.nifti.get_display_affine(t1_nii)
    mrsi_to_display = np.linalg.inv(t1_display_affine).dot(mrsi_affine)
    logger.debug(f"T1 display affine:\n{t1_display_affine}")
    logger.debug(f"MRSI affine:\n{mrsi_affine}")
    logger.debug(f"MRSI to display affine:\n{mrsi_to_display}")
    return mrsi_to_display


def display_nifti(
    nii: spatialimages.SpatialImage | GESeries.NiiBase,
    orientation: tuple[str, str, str] | None = None,
    **kwargs,
) -> None:
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
    None
        See `display_images` -- the image grid is displayed, not returned.
    """
    if isinstance(nii, GESeries.NiiBase):
        nii = nii.nii
    return display_images(utils.nifti.orient_nifti(nii, orientation), **kwargs)


def display_images(
    images: npt.NDArray | list,
    titles: list[str] | None = None,
    fig_title: str = "",
    cmap: str | None = None,
    imshow_kws: dict[str, Any] | None = None,
    colorbar: bool = False,
    colorbar_kws: dict[str, Any] | None = None,
    fig_kws: dict[str, Any] | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    v_percentile: float = 1.0,
    zeros_as_nan: bool = True,
    zooms: tuple[float, float] | None = None,
    panel_height: str | float | None = None,
    grid_max_height: str | float | None = None,
    **kwargs,
) -> None:
    """Display a grid of 2D/3D/4D images, with a slice slider for 3D/4D input.

    The panels are rendered straight to images at the data's own resolution
    and handed to an interactive widget that draws the titles and colorbar
    around them as HTML; no matplotlib `Figure` is involved, which is why
    nothing is returned.

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
        Colormap to use. Defaults to `rcParams["image.cmap"]`.
    imshow_kws : dict, optional
        Accepted and ignored: there is no `imshow` call left to forward them
        to. Kept so existing call sites keep working.
    colorbar : bool, optional
        Whether to display a colorbar, by default False.
    colorbar_kws : dict, optional
        Only the `"mode"` key is read, either `"single"` (one colorbar for
        the grid) or `"each"` (one per panel, autoscaled to that panel and
        slice). Remaining keys are ignored -- the colorbar is HTML now, not a
        matplotlib artist.
    fig_kws : dict, optional
        Accepted and ignored: there is no `Figure` left to configure. Kept so
        existing call sites keep working.
    vmin : float, optional
        Lower display bound. Estimated from the data if omitted.
    vmax : float, optional
        Upper display bound. Estimated from the data if omitted.
    v_percentile : float, optional
        Percentile used to estimate `vmin`/`vmax` when not given, by
        default 1.0.
    zeros_as_nan : bool, optional
        If True, zeros are treated as NaN: kept out of the `vmin`/`vmax`
        estimate so the background cannot drag the scaling down, and rendered
        transparent rather than as the colormap's low end. By default True.
    zooms : tuple[float, float], optional
        Physical (row, col) voxel edge lengths of the displayed slice, e.g.
        from `utils.nifti.get_display_zooms`. When given, panels are rendered
        with an aspect ratio that matches real anatomy instead of one square
        pixel per voxel; the through-plane spacing is irrelevant since only
        one 2D slice is ever on screen at a time. Defaults to the pixel-count
        aspect ratio (square voxels) when omitted.
    panel_height : str or float, optional
        How tall a single panel may grow, as a CSS length (`"30rem"`,
        `"400px"`) or a bare number read as rem. The compact default suits a
        row of low-resolution panels; raise it to read a high-resolution
        anatomical series. Defaults to `rcParams["grid.panel_height"]`.
    grid_max_height : str or float, optional
        How tall the panel box may grow before it scrolls, same units. Worth
        raising alongside `panel_height` for a tall single-column view.
        Defaults to `rcParams["grid.max_height"]`.
    **kwargs
        Accepted and ignored: axis parameters such as `aspect`, `xlabel` or
        `xticks` have no Axes to apply to. Kept so existing call sites keep
        working.

    Returns
    -------
    None
        The grid is displayed as a widget. For a single-slice input the
        widget simply has no slider.
    """
    cmap = resolve_rc(cmap, "image.cmap")
    if titles is None:
        titles = []

    # Convert list of images to 4D numpy array
    if isinstance(images, list):
        images = np.stack(images, axis=-1)

    # Cast images to float32 for display
    images = images.astype(np.float32)
    original_dims = images.ndim

    if original_dims == 2:
        num_images = 1
        num_cols = 1
        slice_idx = 0
        images = images[:, :, np.newaxis, np.newaxis]
    elif original_dims == 3:
        num_images = 1
        num_cols = 1
        # Put the images into a 4D array for easier handling later
        slice_idx = images.shape[2] // 2
        images = images[:, :, :, np.newaxis]
    elif original_dims == 4:
        num_images = images.shape[3]
        # Nominal columns per row. The widget wraps to fewer when the pane is
        # too narrow for them, so this is the *most* a row will ever hold.
        max_cols = rcParams["grid.max_cols"]
        num_cols = min(max_cols, num_images)
        slice_idx = images.shape[2] // 2
        num_rows = -(-num_images // max_cols)
        logger.debug(f"Displaying {num_images} images in a {num_rows}x{num_cols} grid.")
        # Check that 3rd dimension is the same for all images
        for i in range(1, num_images):
            if images.shape[2] != images[:, :, :, i].shape[2]:
                raise ValueError(
                    "All images must have the same number of slices in the 3rd dimension."
                )
    else:
        raise ValueError("Input images must be either 3D or 4D.")

    # Nothing is masked here: `zeros_as_nan` is handed to the two places that
    # actually act on it -- `fast_bounds` for the numbers, `_Layer` for the
    # pixels -- so no copy of the volume is made and pinning `vmin`/`vmax` no
    # longer drags the display mask along with it.
    vmin, vmax = _resolve_display_bounds(
        images, vmin, vmax, v_percentile, exclude_zeros=zeros_as_nan
    )

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

    ignored = {
        name: value
        for name, value in (("imshow_kws", imshow_kws), ("fig_kws", fig_kws), *kwargs.items())
        if value is not None
    }
    if ignored:
        logger.debug(
            f"Ignoring matplotlib-only arguments {sorted(ignored)}: the panels are rendered "
            "directly to images, so there is no Figure or Axes to configure."
        )

    if colorbar:
        colorbar_mode = colorbar_kws.pop("mode", "single") if colorbar_kws is not None else "single"
    else:
        colorbar_mode = ""

    frames, bounds = _render_image_grid_frames(
        images,
        num_images=num_images,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        per_panel_bounds=colorbar_mode == "each",
        zeros_as_nan=zeros_as_nan,
        pixel_aspect=None if zooms is None else zooms[1] / zooms[0],
    )
    size_vars = {
        "--mnu-panel-max-h": validate_css_length(resolve_rc(panel_height, "grid.panel_height")),
        "--mnu-grid-max-h": validate_css_length(resolve_rc(grid_max_height, "grid.max_height")),
        "--mnu-panel-min-h": rcParams["grid.panel_min_height"],
        "--mnu-panel-min-w": rcParams["grid.panel_min_width"],
    }
    ipy_display(
        ImageGridWidget(
            frames=frames,
            bounds=bounds,
            colormap_stops=_colormap_stops(cmap),
            num_cols=num_cols,
            titles=titles,
            fig_title=fig_title,
            colorbar_mode=colorbar_mode,
            initial_index=slice_idx,
            size_vars=size_vars,
        )
    )
    return None


def _colormap_stops(cmap: str, num_stops: int = 64) -> list[str]:
    """Sample a matplotlib colormap into CSS hex stops, low to high.

    The widget's colorbar is a CSS `linear-gradient`, so it needs the colormap
    as colours rather than as a `Colormap`. Sampling it here (rather than
    reimplementing the colormap in JS) is what keeps the bar and the pixels it
    explains provably the same colormap. 64 stops is well past the point where
    the browser's linear interpolation between them is distinguishable from
    the real thing, and costs ~0.5 kB.
    """
    colors = colormaps[cmap](np.linspace(0, 1, num_stops), bytes=True)
    return [f"#{r:02x}{g:02x}{b:02x}" for r, g, b, _ in colors]


class _Layer(NamedTuple):
    """One colormapped volume in a frame, composited over the ones before it.

    `bounds` is either a single `(vmin, vmax)` used for every frame, or one
    pair per job when each frame autoscales to itself. `alpha` is matplotlib's
    `imshow(alpha=...)`: the base layer is opaque, layers above it blend.
    `zeros_as_nan` renders this layer's exact zeros transparent. It is a
    per-layer choice because an overlay wants its all-zero background to drop
    out while the anatomical underneath keeps every voxel it has.
    """

    volume: npt.NDArray
    cmap: str
    bounds: tuple[float, float] | list[tuple[float, float]]
    alpha: float = 1.0
    zeros_as_nan: bool = False


def _stretch_to_pixel_aspect(array: npt.NDArray, pixel_aspect: float) -> npt.NDArray:
    """Nearest-neighbour-upscale one axis so the raster's own pixels are voxel-shaped.

    `pixel_aspect` is physical col-spacing / row-spacing. Only ever upscales
    -- the smaller-per-mm axis grows to match the other -- so no resolution is
    lost on either axis, unlike scaling both to a common isotropic spacing.
    Nearest-neighbour (voxel duplication) rather than interpolation: this is a
    display-only raster, not the analysis data, and duplication is what keeps
    a hard voxel edge hard instead of blurring it.
    """
    rows, cols = array.shape[:2]
    row_scale = max(1.0, 1.0 / pixel_aspect)
    col_scale = max(1.0, pixel_aspect)
    new_rows, new_cols = max(1, round(rows * row_scale)), max(1, round(cols * col_scale))
    if (new_rows, new_cols) == (rows, cols):
        return array
    resized = PIL.Image.fromarray(array).resize((new_cols, new_rows), PIL.Image.NEAREST)
    return np.asarray(resized)


def _render_frames(
    layers: Sequence[_Layer],
    jobs: Sequence[tuple[int, ...]],
    *,
    quality: int,
    lossless: bool = False,
    keep_alpha: bool = False,
    pixel_aspect: float | None = None,
    max_workers: int = 8,
) -> list[bytes]:
    """Render a batch of colormapped rasters to WebP bytes, one per job.

    The single frame renderer behind every widget in this module: the
    `display_images` grid (one opaque layer, one job per (slice, panel)) and
    the MRSI inspector's anatomicals (a gray T1 under a half-transparent magma
    overlay, one job per slice) differ only in their layers and in which axes
    a job indexes. A job is the trailing index into each layer's volume, so
    `volume[:, :, *job]` is the 2-D image that frame colormaps.

    These frames are pure raster -- no ticks, labels, titles, colorbar or
    contours; the widgets rebuild all of that as DOM around them. Going
    through a matplotlib `Figure`/Agg canvas to produce that costs ~130 ms per
    frame, almost all of it rasterization and PNG encoding, for arithmetic
    that is really only `cmap(norm(arr))` plus an alpha blend. So the Figure
    is gone: matplotlib still *supplies* the colormaps (via `ScalarMappable`,
    so nothing is reimplemented and the panels, the colorbar gradient and the
    inspector can't drift apart), and the blend and encode are done directly.

    Frames come out at the volume's native `(nrows, ncols)` rather than a
    dpi-derived size, and the browser scales them to whatever width it has.
    The client side is resolution-independent -- the SVG voxel overlay is
    sized by a data-coordinate `viewBox` and clicks are mapped through the
    rendered `rect.width` -- so this is purely sharper.

    WebP rather than PNG: at native resolution an anatomical frame is ~318 kB
    as PNG but ~42 kB as WebP q92 (41 dB PSNR against the exact composite),
    which is what keeps a widget's payload manageable. Pass `lossless` for
    small hard-edged images, where the ringing shows and the payload argument
    disappears. `keep_alpha` carries the colormap's "bad" pixels (NaN, so also
    whatever a layer's `zeros_as_nan` masked out) through as transparent instead
    of flattening them onto a baked-in background.
    `pixel_aspect`, when given, upscales one axis of the composite (see
    `_stretch_to_pixel_aspect`) so the frame's own dimensions already encode
    the physical voxel aspect ratio -- the browser then needs no separate
    aspect correction and stays on the same natural-size sizing path as the
    unscaled case.

    WebP encoding releases the GIL, so frames are encoded on a thread pool.
    """
    # Layers scaled the same way for every frame get one `ScalarMappable`
    # for the whole batch; per-job bounds have to build theirs per frame.
    shared = [
        None
        if isinstance(layer.bounds, list)
        else ScalarMappable(Normalize(*layer.bounds), layer.cmap)
        for layer in layers
    ]

    def render(index: int) -> bytes:
        selector = (slice(None), slice(None), *jobs[index])
        composite: npt.NDArray | None = None
        for layer, mappable in zip(layers, shared):
            if mappable is None:
                bounds = layer.bounds[index]  # pyright: ignore[reportIndexIssue]
                mappable = ScalarMappable(Normalize(*bounds), layer.cmap)
            data = layer.volume[selector]
            if layer.zeros_as_nan:
                # Per frame, not per volume: a 2-D `where` on the slice about
                # to be drawn costs less than a copy of the whole volume.
                data = np.where(data == 0, np.nan, data)
            rgba = mappable.to_rgba(data, bytes=True)
            if composite is None:
                composite = rgba
                continue
            # Reproduce matplotlib's compositing of a second `imshow(alpha=)`
            # over the first: the layer's alpha is scaled by the overlay's own
            # per-pixel alpha, so pixels the colormap marks "bad" (NaN, alpha
            # 0) let the layer below through untouched instead of being
            # blended toward the bad colour.
            weight = (rgba[..., 3:4].astype(np.uint16) * round(layer.alpha * 255)) // 255
            composite[..., :3] = (
                composite[..., :3] * (255 - weight) + rgba[..., :3] * weight
            ) // 255

        assert composite is not None, "at least one layer is required"
        if pixel_aspect is not None:
            composite = _stretch_to_pixel_aspect(composite, pixel_aspect)
        if keep_alpha:
            return _encode_webp(composite, "RGBA", quality=quality, lossless=lossless)
        return _encode_webp(
            np.ascontiguousarray(composite[..., :3]), quality=quality, lossless=lossless
        )

    logger.debug(f"Rendering {len(jobs)} frames on {max_workers} threads")
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(render, range(len(jobs))))


def _panel_bounds(panel: npt.NDArray, zeros_as_nan: bool = False) -> tuple[float, float]:
    """Autoscale one panel the way `AxesImage.autoscale` did: to its own extent.

    Guards the two cases matplotlib's `Normalize` would otherwise have to
    absorb: an all-NaN panel (nothing to scale to) and a constant one (a zero
    -width range maps every pixel to the colormap's low end). `zeros_as_nan`
    keeps a zero background out of that extent, matching what the layer will
    then draw; an all-zero panel falls back to the zeros so the pair stays
    finite.
    """
    if zeros_as_nan and np.any(np.isfinite(panel) & (panel != 0)):
        panel = np.where(panel == 0, np.nan, panel)
    if not np.isfinite(panel).any():
        return 0.0, 1.0
    return _widen_if_flat(float(np.nanmin(panel)), float(np.nanmax(panel)))


def _render_image_grid_frames(
    images: npt.NDArray,
    *,
    num_images: int,
    cmap: str,
    vmin: float | None,
    vmax: float | None,
    per_panel_bounds: bool,
    zeros_as_nan: bool = False,
    pixel_aspect: float | None = None,
    quality: int = GRID_FRAME_QUALITY,
    max_workers: int = 8,
) -> tuple[list[list[bytes]], list[list[tuple[float, float]]]]:
    """Shape a `display_images` grid for `_render_frames`: one job per panel.

    Rasterising is entirely `_render_frames`' job; what is grid-specific is
    the bounds each panel is normalised with -- one shared pair, or, under
    `colorbar_kws={"mode": "each"}`, one per panel and slice -- and the
    slice-major nesting `_widgets.ImageGridWidget` indexes frames by. The
    bounds are returned alongside because the widget prints them on its
    colorbar.

    Panels are rendered with alpha: values the colormap marks "bad" (NaN, so
    also everything `zeros_as_nan` masked out) stay transparent and take the
    panel's background in both light and dark themes instead of a baked-in
    figure colour. Small panels are encoded losslessly -- lossy WebP's ringing
    is invisible on a 512-row anatomical and obvious on a 20x20 mask, where
    the payload argument for lossy disappears anyway (see
    `GRID_LOSSLESS_MAX_PIXELS`).
    """
    n_slices = images.shape[2]
    if per_panel_bounds:
        bounds = [
            [_panel_bounds(images[:, :, s, p], zeros_as_nan) for p in range(num_images)]
            for s in range(n_slices)
        ]
    else:
        shared = _widen_if_flat(
            0.0 if vmin is None else float(vmin), 1.0 if vmax is None else float(vmax)
        )
        bounds = [[shared] * num_images for _ in range(n_slices)]

    jobs = [(s, p) for s in range(n_slices) for p in range(num_images)]
    rendered = _render_frames(
        [_Layer(images, cmap, [bounds[s][p] for s, p in jobs], zeros_as_nan=zeros_as_nan)],
        jobs,
        quality=quality,
        lossless=images.shape[0] * images.shape[1] <= GRID_LOSSLESS_MAX_PIXELS,
        keep_alpha=True,
        pixel_aspect=pixel_aspect,
        max_workers=max_workers,
    )
    return [rendered[s * num_images : (s + 1) * num_images] for s in range(n_slices)], bounds


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


def _render_overlay_raster_frames(
    t1_images: npt.NDArray,
    data_images: npt.NDArray,
    mask: npt.NDArray[np.bool_] | None,
    only_overlay: bool,
    cmap: str | None = None,
    alpha: float = 0.5,
    vmin: float | None = None,
    vmax: float | None = None,
    v_percentile: float = 1.0,
    zeros_as_nan: bool = True,
) -> tuple[list[list[bytes]], list[list[tuple[float, float]]], list[str]]:
    """Render `overlay_image_data_on_T1`'s panels straight to raster, one job per slice.

    Reuses `_render_frames` exactly as `inspect_MRSI_spectra` already does --
    a gray T1 layer, a colormapped data layer, alpha-blended for the overlay
    panel -- rather than a matplotlib `Figure`/`savefig` round-trip. `mask`
    (fill, non-contour) is applied the same way the matplotlib engine applies
    it: `np.where(mask, data, nan)` before the data layer is built. Panel
    bounds are computed the same way `_draw_single_axis_overlay` computes
    them, for visual parity between engines.

    `zeros_as_nan` defaults to True and applies to the *data* layers only: an
    overlay's zeros are background, and painting them with the colormap's low
    end would hide the anatomical underneath. The T1 layer keeps its zeros --
    they are the head's surroundings, and dropping them out would put the page
    background inside the skull outline.
    """
    cmap = resolve_rc(cmap, "image.cmap")
    n_slices = t1_images.shape[2]
    jobs = [(s,) for s in range(n_slices)]

    anat_bounds = fast_bounds(t1_images, p_lower=1.0, p_upper=99.0)
    masked_data = np.where(mask, data_images, np.nan) if mask is not None else data_images
    data_bounds = _resolve_display_bounds(
        masked_data, vmin, vmax, v_percentile, exclude_zeros=zeros_as_nan
    )

    panels: list[tuple[str, list[_Layer], tuple[float, float]]] = []
    if not only_overlay:
        panels.append(("T1 Image", [_Layer(t1_images, "gray", anat_bounds)], anat_bounds))
        panels.append(
            (
                "Resampled Data",
                [_Layer(masked_data, cmap, data_bounds, zeros_as_nan=zeros_as_nan)],
                data_bounds,
            )
        )
    panels.append(
        (
            "Overlay",
            [
                _Layer(t1_images, "gray", anat_bounds),
                _Layer(masked_data, cmap, data_bounds, alpha=alpha, zeros_as_nan=zeros_as_nan),
            ],
            data_bounds,
        )
    )

    # Same encoder rule as the `display_images` grid: without it a small
    # panel here came out lossy while the identical array rendered losslessly
    # there, and WebP's ringing on hard voxel edges made the two disagree.
    lossless = t1_images.shape[0] * t1_images.shape[1] <= GRID_LOSSLESS_MAX_PIXELS
    rendered = [
        _render_frames(layers, jobs, quality=GRID_FRAME_QUALITY, lossless=lossless, keep_alpha=True)
        for _, layers, _ in panels
    ]
    frames = [[rendered[p][s] for p in range(len(panels))] for s in range(n_slices)]
    bounds = [[panel_bounds for _, _, panel_bounds in panels] for _ in range(n_slices)]
    titles = [title for title, _, _ in panels]
    return frames, bounds, titles


@overload
def overlay_image_data_on_T1(
    t1_images: npt.NDArray,
    data_images: npt.NDArray | None = None,
    *,
    ax: Axes,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    mask_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[list[AxesImage], QuadContourSet | None]: ...
@overload
def overlay_image_data_on_T1(
    t1_images: npt.NDArray,
    data_images: npt.NDArray,
    *,
    engine: Literal["matplotlib"],
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    only_overlay: bool = False,
    mask_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]] | None: ...
@overload
def overlay_image_data_on_T1(
    t1_images: npt.NDArray,
    data_images: npt.NDArray,
    *,
    engine: Literal["raster"] | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    only_overlay: bool = False,
    mask_kwargs: dict[str, Any] | None = None,
    **kwargs,
) -> None: ...
def overlay_image_data_on_T1(
    t1_images: npt.NDArray,
    data_images: npt.NDArray | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    only_overlay: bool = False,
    mask_kwargs: dict[str, Any] | None = None,
    engine: Literal["raster", "matplotlib"] | None = None,
    ax: Axes | None = None,
    **kwargs,
) -> tuple[Figure, list[Axes]] | tuple[list[AxesImage], QuadContourSet | None] | None:
    """Display T1, resampled data, and overlay images side by side with a slice slider.

    Parameters
    ----------
    t1_images : ndarray
        T1-weighted anatomical image data.
    data_images : ndarray, optional
        Data image, resampled into T1 space, to overlay. Required unless `ax`
        is given for a T1-only draw.
    mask : ndarray, optional
        Boolean mask applied to the overlay data, by default None.
    mask_contour : ndarray or bool, optional
        Mask (or `True` to reuse `mask`) whose contour is drawn on the
        overlay axis. Matplotlib engine only -- the raster engine has no
        cheap contour equivalent and warns and ignores it.
    only_overlay : bool, optional
        If True, only show the overlay axis instead of T1, data, and
        overlay side by side, by default False.
    mask_kwargs : dict, optional
        Keyword arguments customizing the mask contour's appearance.
    engine : {"raster", "matplotlib"}, optional
        Rendering path. Defaults to `"raster"` (fast, no `Figure`) unless
        `ax` is given, which forces matplotlib -- there is no client-side
        `Axes` for a raster widget to draw into. If both are given, `ax`
        wins and a mismatched `engine="raster"` is ignored with a warning.
    ax : matplotlib.axes.Axes, optional
        Draw a single overlay panel onto this axis instead of displaying the
        T1/data/overlay widget. Implies the matplotlib engine.
    **kwargs
        Additional keyword arguments forwarded to the resolved rendering path.

    Returns
    -------
    tuple[Figure, list[Axes]] or tuple[list[AxesImage], QuadContourSet | None] or None
        `ax` given: the plotted images and contours. `engine="matplotlib"`
        with no `ax`: `(Figure, axes)` for single-slice input, `None` (widget
        displayed) for multi-slice. Raster engine (default): always `None`
        (widget displayed), single-slice included.
    """
    if ax is not None:
        if engine == "raster":
            logger.warning(
                "`ax` was given, which requires the matplotlib engine; ignoring engine='raster'."
            )
        return _draw_single_axis_overlay(
            t1_images,
            data_images,
            mask=mask,
            mask_contour=mask_contour,
            ax=ax,
            mask_kwargs=mask_kwargs,
            **kwargs,
        )

    if data_images is None:
        raise ValueError("data_images is required unless ax is given for a T1-only draw.")
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

    if engine == "matplotlib":
        # `mask_contour` only draws a contour when it's `True` (reuse `mask`); an
        # explicit array is not applied here (matches this function's long-standing
        # behavior; `_draw_single_axis_overlay` is the one that honours it).
        resolved_mask_contour = mask if mask_contour is True else None
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
            ipy_display(
                SliceViewerWidget(frames=frames, n_slices=n_slices, initial_index=slice_idx)
            )
            return None

        fig = plt.figure(figsize=(12, 4))
        axes = [fig.subplots(1, 1)] if only_overlay else list(fig.subplots(1, 3))
        _draw_overlay_slice(fig, axes, t1_images, data_images, slice_idx, **overlay_kwargs)
        return fig, axes

    if mask_contour is not None and mask_contour is not False:
        logger.warning(
            "mask_contour has no raster engine equivalent (no contour tracing in "
            "_render_frames); ignoring it. Pass engine='matplotlib' to draw a contour."
        )

    raster_kwarg_names = {"cmap", "alpha", "vmin", "vmax", "v_percentile", "zeros_as_nan"}
    raster_kwargs = {k: v for k, v in kwargs.items() if k in raster_kwarg_names}
    ignored = {k: v for k, v in kwargs.items() if k not in raster_kwarg_names}
    if ignored:
        logger.debug(
            f"Ignoring matplotlib-only arguments {sorted(ignored)} on the raster engine: "
            "there is no Axes to configure."
        )

    frames, bounds, titles = _render_overlay_raster_frames(
        t1_images, data_images, mask, only_overlay, **raster_kwargs
    )
    ipy_display(
        ImageGridWidget(
            frames=frames,
            bounds=bounds,
            colormap_stops=_colormap_stops(resolve_rc(raster_kwargs.get("cmap"), "image.cmap")),
            num_cols=len(titles),
            titles=titles,
            initial_index=slice_idx,
        )
    )
    return None


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
    """Draw one slice of the `overlay_image_data_on_T1` panels onto given axes.

    The data-only panel draws the data as a *base* layer, where
    `_draw_single_axis_overlay`'s `zeros_as_nan` does not reach (it masks the
    overlay layer). So the mask is applied here instead, keeping this panel
    matching both the overlay panel beside it and the raster engine's.
    """
    axes_idx = 0
    if not only_overlay:
        _draw_single_axis_overlay(
            t1_images[:, :, slice_idx], ax=axes[axes_idx], cmap="gray", **kwargs
        )
        axes[axes_idx].set_title("T1 Image")
        axes_idx += 1
        data_slice = data_images[:, :, slice_idx]
        if kwargs.get("zeros_as_nan", True):
            data_slice = np.where(data_slice == 0, np.nan, data_slice)
        _draw_single_axis_overlay(data_slice, ax=axes[axes_idx], **kwargs)
        axes[axes_idx].set_title("Resampled Data")
        axes_idx += 1

    _draw_single_axis_overlay(
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


def _draw_single_axis_overlay(
    base_image: npt.NDArray,
    overlay_image: npt.NDArray | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
    mask_contour: npt.NDArray[np.bool_] | bool | None = None,
    *,
    ax: Axes,
    cmap: str | None = None,
    alpha: float = 0.5,
    mask_kwargs: dict[str, Any] | None = None,
    zeros_as_nan: bool = True,
    **kwargs,
) -> tuple[list[AxesImage], QuadContourSet | None]:
    """Draw one anatomical image, optionally with a data overlay/mask, onto `ax`.

    The folded body of the former `overlay_image_data_on_T1_on_ax` --
    `overlay_image_data_on_T1(..., ax=...)` dispatches straight here. Real
    matplotlib is unavoidable on this path: there is no client-side `Axes` for
    a raster widget to draw into.

    `zeros_as_nan` masks the overlay's zeros so they render transparent and
    scale is taken from real signal only -- the same default, and the same
    layer-by-layer meaning, as the raster engine's `_Layer(zeros_as_nan=...)`,
    so the two engines produce the same picture.
    """
    cmap = resolve_rc(cmap, "image.cmap")
    # Copy default image params and update with any provided kwargs
    remove_keys = ["vmin", "vmax", "v_percentile"]
    # Adjust aspect ratio based on image shape
    aspect_ratio = base_image.shape[1] / base_image.shape[0]
    kwargs.setdefault("aspect", aspect_ratio)
    logger.trace(f"Setting image aspect ratio to {kwargs['aspect']:.2f}.")
    image_params = rcParams.group("image.ax")
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
        return im, None

    vmin, vmax = _resolve_display_bounds(
        overlay_image,
        kwargs.pop("vmin", None),
        kwargs.pop("vmax", None),
        kwargs.pop("v_percentile", 1.0),
        exclude_zeros=zeros_as_nan,
    )

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
    if zeros_as_nan:
        overlay_image = np.where(overlay_image == 0, np.nan, overlay_image)

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
        contour_kwargs = rcParams.group("image.mask")
        if mask_kwargs is not None:
            contour_kwargs.update(mask_kwargs)
        logger.debug(f"Applying mask contour with parameters: {contour_kwargs}")
        contours = ax.contour(mask_contour, **contour_kwargs)
    else:
        contours = None

    ax.set(**image_params)
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
        If True (and `magnitude` is False), phase each voxel independently
        before display, by default True. See `mrsi_spectra_for_display` for
        why this is done per voxel rather than through a single global fit.
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

    mrsi_to_display = _mrsi_to_display_affine(T1.nii, MRSI)
    display_to_mrsi = np.linalg.inv(mrsi_to_display)

    nx, ny, n_mrsi_slices = MRSI.spec.shape[1], MRSI.spec.shape[2], MRSI.spec.shape[3]
    initial_voxel = (nx // 2, ny // 2, n_mrsi_slices // 2)

    anat_vmin, anat_vmax = fast_bounds(t1_images)
    mrsi_vmin, mrsi_vmax = fast_bounds(mrsi_images, exclude_zeros=True)
    logger.debug(f"Anatomical image bounds for display: vmin={anat_vmin:4g}, vmax={anat_vmax:4g}")
    logger.debug(f"MRSI image bounds for display: vmin={mrsi_vmin:4g}, vmax={mrsi_vmax:4g}")

    def mrsi_slice_for(anat_slice_idx: int) -> int:
        raw = affines.apply_affine(display_to_mrsi, [0, 0, anat_slice_idx])[2]
        return int(np.clip(np.round(raw), 0, n_mrsi_slices - 1))

    logger.debug("Starting rendering of Left frames")

    left_frames = _render_frames(
        [
            _Layer(t1_images, "gray", (anat_vmin, anat_vmax)),
            _Layer(mrsi_images, "magma", (mrsi_vmin, mrsi_vmax), alpha=0.5, zeros_as_nan=True),
        ],
        [(s,) for s in range(n_anat_slices)],
        quality=MRSI_FRAME_QUALITY,
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

    ppm_axis, spectra_array = mrsi_spectra_for_display(
        MRSI, magnitude=magnitude, autophase=autophase
    )
    npts = ppm_axis.size
    spectra_bytes, spectra_scale = _encode_spectra(spectra_array)
    # Bounds over the whole grid, computed here rather than in the browser:
    # they set the travel of the widget's manual y-limit slider, which has to
    # be drawable before anything has scanned the 2.9M-value buffer.
    spectra_min = float(np.nanmin(spectra_array)) if spectra_array.size else 0.0
    spectra_max = float(np.nanmax(spectra_array)) if spectra_array.size else 1.0

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
            ppm=ppm_axis.tolist(),
            spectra_bytes=spectra_bytes,
            spectra_scale=spectra_scale,
            spectra_min=spectra_min,
            spectra_max=spectra_max,
            npts=npts,
            spectrum_label=spectrum_label,
        )
    )


def mrsi_spectra_for_display(
    MRSI: GESeries.MRSISeries,
    *,
    magnitude: bool = False,
    autophase: bool = True,
) -> tuple[npt.NDArray, npt.NDArray]:
    """Build the exact ppm axis and spectra grid `inspect_MRSI_spectra` displays.

    Split out of the widget so the displayed numbers can be inspected directly
    — the inspector ships precisely this array, so plotting `spectra[x, y, z]`
    against `ppm` reproduces a voxel's panel exactly:

    >>> ppm, spectra = mrsi_spectra_for_display(mrsi)  # doctest: +SKIP
    >>> plt.plot(ppm, spectra[8, 8, 8])                # doctest: +SKIP

    Parameters
    ----------
    MRSI : GESeries.MRSISeries
        The MRSI series to read spectra from.
    magnitude : bool, optional
        If True, return the magnitude spectrum, by default False.
    autophase : bool, optional
        If True (and `magnitude` is False), phase each voxel independently,
        by default True.

    Returns
    -------
    ppm : ndarray
        1-D ppm axis, spanning the full acquired sweep.
    spectra : ndarray
        Real-valued grid indexed `[x, y, mrsi_slice, point]`, matching the
        voxel coordinates the widget's picker reports.
    """
    ppm = np.asarray(MRSI.ppm.values, dtype=float)
    spectral_dim = MRSI.spec.dims[0]
    # Top left of the image is (0, 0); the MRSI grid's coordinate system
    # starts bottom-right, so the array is reversed along i/j and the (x, y)
    # voxel axes read from (j, i) rather than (i, j) -- matches the mapping
    # `overlay_voxel_on_T1`/the widget's client-side voxel picker expect.
    grid = (
        MRSI.spec.isel(i=slice(None, None, -1), j=slice(None, None, -1))
        .transpose("j", "i", "k", spectral_dim)
        .values
    )

    if magnitude:
        return ppm, np.abs(grid)
    if not autophase:
        return ppm, grid.real
    # Phase every voxel by its own peak instead of applying one global
    # correction to all of them. xmris' `autophase(mode="single")` derives a
    # single zero-order phase from the grid's strongest voxel; on real data
    # that leaves the median voxel only ~0.52 absorptive and outright inverts
    # ~29% of them, which reads as corrupted spectra. Rotating each voxel by
    # the argument of its own largest point makes every peak absorptive and
    # positive, is one complex multiply (~0.02 s for a 16x16x16 grid, against
    # ~0.85 s for the global fit), and is display-only -- no fitting decision
    # is taken here. `mode="all"` upstream would be the real home for this;
    # it currently raises NotImplementedError.
    peak = np.abs(grid).argmax(axis=-1)
    xs, ys, zs = np.indices(peak.shape)
    phi = np.angle(grid[xs, ys, zs, peak])
    return ppm, (grid * np.exp(-1j * phi)[..., None]).real


def _encode_spectra(spectra_array: npt.NDArray) -> tuple[bytes, float]:
    """Pack the per-voxel spectra grid into a compact wire buffer.

    Returns the zlib-compressed float16 buffer and the scale factor needed to
    recover the original values (`value = stored * scale`).

    Two reductions over the obvious `astype(np.float32).tobytes()`, which for a
    16x16x16 grid at 700 points is 11.5 MB before base64 inflates it to 15.3:

    - **float16 halves it.** This buffer only ever feeds a line plot, so the
      ~3 significant digits it keeps are far inside display precision. Raw
      spectra do overflow float16's 65504 limit, though, so the grid is scaled
      into range first and the factor travels with it.
    - **zlib then takes another ~8x**, because neighbouring points in a
      spectrum are highly correlated. The browser inflates it with the built-in
      `DecompressionStream`, so this costs no JS dependency.
    """
    nbytes = spectra_array.size * 4
    if nbytes > _SPECTRA_WARN_BYTES:
        logger.warning(
            f"MRSI spectra buffer is large ({nbytes / 1024**2:.0f} MB as float32 for a "
            f"{'x'.join(str(d) for d in spectra_array.shape)} grid). The widget embeds it in "
            "its output, so the notebook and any built docs page will grow accordingly."
        )
    peak = float(np.nanmax(np.abs(spectra_array))) if spectra_array.size else 0.0
    # A power of two, not `peak / target`: scaling by one shifts every value's
    # exponent and leaves its mantissa bits untouched, so it is both exact (no
    # rounding on top of the float16 cast) and keeps the buffer as compressible
    # as it was. An arbitrary divisor perturbs every mantissa and cost ~4x on
    # the compressed size when measured.
    scale = 2.0 ** np.ceil(np.log2(peak / _F16_TARGET_MAX)) if peak > _F16_TARGET_MAX else 1.0
    scaled = (spectra_array / scale) if scale != 1.0 else spectra_array
    return zlib.compress(scaled.astype(np.float16).tobytes(), 6), float(scale)


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

    mrsi_to_display = _mrsi_to_display_affine(t1.nii, MRSI)

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
    default_overlay_kwargs: dict[str, Any] = {"cmap": rcParams["image.cmap"], "alpha": 0.5}

    # Update default kwargs with provided kwargs
    img_vmin, img_vmax = fast_bounds(t1_images[:, :, slice_idx])
    default_image_kwargs.update({"vmin": img_vmin, "vmax": img_vmax})
    image_kwargs = default_image_kwargs | (image_kwargs or {})

    ax.imshow(t1_images[:, :, slice_idx], origin="upper", **image_kwargs)
    logger.trace(f"Sum of t1 image slice: {np.nansum(t1_images[:, :, slice_idx])}")
    if show_overlay:
        overlay_slice = mrsi_images[:, :, slice_idx]
        mrsi_vmin, mrsi_vmax = fast_bounds(overlay_slice, exclude_zeros=True)
        overlay_kwargs = default_overlay_kwargs | (overlay_kwargs or {})
        overlay_kwargs.update({"vmin": mrsi_vmin, "vmax": mrsi_vmax})
        # Zeros here are outside the MRSI FOV, not signal: masked so the
        # anatomical shows through instead of a magma-black wash.
        ax.imshow(
            np.where(overlay_slice == 0, np.nan, overlay_slice),
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
