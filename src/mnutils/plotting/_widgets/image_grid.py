import base64
from pathlib import Path
from typing import Any

from ._html import render_html
from ._shared import load_css, load_esm

_DIR = Path(__file__).parent
_JS = load_esm(_DIR / "image_grid.js")
_CSS = load_css(_DIR / "image_grid.css")


class ImageGridWidget:
    """Grid of pre-rendered image panels with one shared slice slider.

    The panels arrive as bare rasters -- no titles, no colorbar, no axes --
    because they were produced by `plotting.images` straight from PIL rather
    than from a matplotlib `Figure`. Everything that used to be drawn *around*
    the pixels is rebuilt here as DOM: the figure title as a heading, each
    panel's title as a `<figcaption>`, and the colorbar as a CSS gradient
    strip with text ticks. That keeps the text sharp at any zoom and lets a
    slice change rewrite a couple of text nodes instead of a whole raster.

    Like every widget in this package the payload is baked in at construction,
    so the slider keeps working in a statically built docs page with no kernel
    behind it.

    Parameters
    ----------
    frames : list[list[bytes]]
        Encoded panel images, slice-major: ``frames[slice_idx][panel_idx]``.
    bounds : list[list[tuple[float, float]]]
        The ``(vmin, vmax)`` the matching frame was normalised with, same
        indexing as `frames`. Only read when a colorbar is shown, but always
        per slice, since ``colorbar_mode="each"`` autoscales every panel on
        every slice.
    colormap_stops : list[str]
        The colormap sampled to CSS hex colors, low to high; interpolated into
        a `linear-gradient` for the colorbar strip.
    num_cols : int
        Panels per row.
    titles : list[str]
        Per-panel captions. May be shorter than the number of panels (a panel
        without a title simply gets none).
    fig_title : str
        Heading above the whole grid; omitted when empty.
    colorbar_mode : str
        ``"single"`` for one strip under the grid, ``"each"`` for a strip per
        panel, ``""`` for none.
    initial_index : int, optional
        Slice shown on first display, by default 0.
    slice_label : str, optional
        Label next to the slider, by default ``"Slice"``.
    size_vars : dict[str, str], optional
        CSS custom properties set on the widget's root element, e.g.
        ``{"--mnu-panel-max-h": "30rem"}``. `plotting.images` fills these from
        the ``grid.*`` rcParams; omitted, the stylesheet's own values apply.
        Values must already be validated CSS lengths -- they land in a `style`
        attribute (see `rcparams.validate_css_length`).
    """

    def __init__(
        self,
        frames: list[list[bytes]],
        bounds: list[list[tuple[float, float]]],
        colormap_stops: list[str],
        num_cols: int,
        titles: list[str] | None = None,
        fig_title: str = "",
        colorbar_mode: str = "",
        initial_index: int = 0,
        slice_label: str = "Slice",
        size_vars: dict[str, str] | None = None,
    ) -> None:
        self._data: dict[str, Any] = {
            "frames": [
                [base64.b64encode(panel).decode("ascii") for panel in slice_frames]
                for slice_frames in frames
            ],
            "bounds": [[list(panel) for panel in slice_bounds] for slice_bounds in bounds],
            "colormap_stops": colormap_stops,
            "n_slices": len(frames),
            "num_panels": len(frames[0]) if frames else 0,
            "num_cols": num_cols,
            "titles": titles if titles else [],
            "fig_title": fig_title,
            "colorbar_mode": colorbar_mode,
            "initial_index": initial_index,
            "slice_label": slice_label,
            "size_vars": size_vars or {},
        }

    def _repr_html_(self) -> str:
        return render_html(_JS, _CSS, self._data)
