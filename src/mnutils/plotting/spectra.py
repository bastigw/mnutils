from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from xmris import DIMS

from ..rcparams import rcParams


def _spectra_ax_params() -> dict[str, Any]:
    """The `spectra.*` rcParams that are `Axes.set` keywords, as such a mapping.

    `rcParams.group("spectra")` would also hand back the ticker keys, which
    `Axes.set` has no idea what to do with -- hence the explicit list.
    """
    return {
        "xlim": rcParams["spectra.xlim"],
        "xlabel": rcParams["spectra.xlabel"],
        "ylabel": rcParams["spectra.ylabel"],
    }


def set_default_ticks(ax: Axes) -> None:
    """Set default tick parameters for a given Axes object."""
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=rcParams["spectra.yticker_bins"],
            steps=rcParams["spectra.ticker_steps"],
        )
    )
    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=rcParams["spectra.xticker_bins"],
            steps=rcParams["spectra.ticker_steps"],
        )
    )
    # By default override stylesheets behaviour of hiding ticks and set bottom and left to true
    ax.tick_params(bottom=True, left=True)


def set_default_spectra_ax_params(ax: Axes, **kwargs) -> None:
    """Set default spectra axis parameters for a given Axes object."""
    spectra_params = _spectra_ax_params()
    spectra_params.update(kwargs)
    ax.set(**spectra_params)


def _resolve_spectra_input(
    data: xr.DataArray | None,
    ppm: npt.NDArray[np.floating] | None,
    spectra: npt.NDArray[np.floating] | None,
    labels: list[str] | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], list[str] | None]:
    """Resolve ppm/spectra arrays and labels from a DataArray or manual fallback args."""
    if data is not None and DIMS.chemical_shift in data.dims:
        ppm = data.coords[DIMS.chemical_shift].values
        spectra = data.values
        if labels is None:
            other_dims = [d for d in data.dims if d != DIMS.chemical_shift]
            if other_dims:
                labels = [str(v) for v in data.coords[other_dims[0]].values]
        return ppm, spectra, labels

    if ppm is None or spectra is None:
        raise ValueError(
            "Provide either `data` (an xr.DataArray with a "
            f"'{DIMS.chemical_shift}' dim) or both `ppm` and `spectra`."
        )
    return ppm, spectra, labels


def plot_spectra(
    data: xr.DataArray | None = None,
    ppm: npt.NDArray[np.floating] | None = None,
    spectra: npt.NDArray[np.floating] | None = None,
    ax: Axes | None = None,
    labels: list[str] | None = None,
    line_kwargs: dict | list[dict] | None = None,
    **kwargs,
) -> Axes:
    """Plot one or more spectra against a chemical-shift (ppm) axis.

    Pass already-phased data — either `data` (an `xr.DataArray` with a
    `chemical_shift` dim/coord, e.g. from `MRSSeries.phase_avg_spec` or
    `.xmr.phase()`/`.xmr.autophase()`) or, as a manual fallback, both `ppm`
    and `spectra` as plain arrays. This function does no phasing itself.
    """
    resolved_ppm, resolved_spectra, resolved_labels = _resolve_spectra_input(
        data, ppm, spectra, labels
    )
    if resolved_labels is None:
        resolved_labels = []

    if ax is None:
        _, ax = plt.subplots()

    spectra_arr = np.asarray(resolved_spectra)
    if spectra_arr.ndim == 1:
        spectra_arr = spectra_arr[np.newaxis, :]  # Convert to 2D for consistency
    if np.iscomplexobj(spectra_arr):
        spectra_arr = spectra_arr.real
    spectra_arr = spectra_arr.astype(np.float64)

    if isinstance(line_kwargs, list):
        if len(line_kwargs) != spectra_arr.shape[0]:
            raise ValueError("Length of line_kwargs list must match number of spectra.")
    elif isinstance(line_kwargs, dict):
        line_kwargs = [line_kwargs] * spectra_arr.shape[0]

    lines = []
    for spec_idx in range(spectra_arr.shape[0]):
        kwargs_to_use = {} if line_kwargs is None else line_kwargs[spec_idx]
        label = (
            resolved_labels[spec_idx]
            if spec_idx < len(resolved_labels)
            else f"Spectrum {spec_idx + 1}"
        )
        (line,) = ax.plot(
            resolved_ppm,
            spectra_arr[spec_idx],
            label=label,
            **kwargs_to_use,
        )
        lines.append(line)

    set_default_ticks(ax)
    set_default_spectra_ax_params(ax, **kwargs)

    if spectra_arr.shape[0] > 1 or resolved_labels:
        ax.legend()

    return ax


def _resolve_fid_input(
    data: xr.DataArray | None,
    time: npt.NDArray[np.floating] | None,
    fids: npt.NDArray[np.complexfloating | np.floating] | None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.complexfloating | np.floating], str]:
    """Resolve a time axis (in ms) and fid values from a DataArray or manual fallback args."""
    if data is not None and DIMS.time in data.dims:
        time_axis = data.coords[DIMS.time].values * 1e3  # s -> ms
        return time_axis, data.values, "Time [ms]"

    if time is None or fids is None:
        raise ValueError(
            f"Provide either `data` (an xr.DataArray with a '{DIMS.time}' dim) "
            "or both `time` and `fids`."
        )
    return time, fids, "Time [ms]"


def plot_fid(
    data: xr.DataArray | None = None,
    time: npt.NDArray[np.floating] | None = None,
    fids: npt.NDArray[np.complexfloating | np.floating] | None = None,
    ax: Axes | None = None,
    labels: list[str] | None = None,
    show_real: bool = True,
    show_imag: bool = True,
    **kwargs,
) -> Axes:
    """Plot one or more FIDs against a time axis.

    Pass either `data` (an `xr.DataArray` with a `time` dim/coord, e.g.
    `RawMRISeries.fids`) or, as a manual fallback, both `time` (in ms) and
    `fids` as plain arrays. This function does no processing itself — build
    the time axis you want to see on the DataArray/array before calling.
    """
    if labels is None:
        labels = []

    xaxis, fid_values, xaxis_label = _resolve_fid_input(data, time, fids)
    fid_values = np.asarray(np.copy(fid_values))
    if fid_values.ndim == 1:
        fid_values = fid_values[np.newaxis, :]  # Convert to 2D for consistency

    if ax is None:
        _, ax = plt.subplots()

    for i in range(fid_values.shape[0]):
        label_real = f"{labels[i]} (Real)" if i < len(labels) else f"FID {i + 1} (Real)"
        label_imag = f"{labels[i]} (Imag)" if i < len(labels) else f"FID {i + 1} (Imag)"
        if show_real:
            ax.plot(xaxis, np.real(fid_values[i, :]), label=label_real)
        if show_imag and np.iscomplexobj(fid_values):
            ax.plot(xaxis, np.imag(fid_values[i, :]), label=label_imag, linestyle="--")

    ax.legend()
    ax.set(**kwargs)

    ax.set_xlabel(xaxis_label)
    ax.set_ylabel("FID signal [a.u.]")

    set_default_ticks(ax)

    sns.set_style("ticks")
    sns.set_palette("colorblind")
    sns.despine()

    return ax


def annotate_metabolites(ax: Axes | None = None, freqs: dict | None = None) -> Axes:
    """Annotate metabolite labels on a plot at specific ppm positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes, optional
        The axes to annotate. Defaults to the current axes.
    freqs : dict, optional
        Mapping of metabolite name to ppm position. Defaults to a predefined
        set of common metabolites (DHO, Glu, Glx, Lac).

    Returns
    -------
    matplotlib.axes.Axes
        The annotated axes.
    """
    if ax is None:
        ax = plt.gca()

    if freqs is None:
        freqs = {
            "DHO": 4.68,
            "Glu": 3.7,
            "Glx": 2.3,
            "Lac": 1.35,
            # "Lip": 0.9,
        }

    # Get lines and find max value at specific ppm positions
    data = np.array([line.get_ydata() for line in ax.get_lines()])
    ppm_axis = np.array([line.get_xdata() for line in ax.get_lines()])[0]

    # Get y limits of plot
    ylims = ax.get_ylim()
    label_offset = (ylims[1] - ylims[0]) * 0.1

    # Plot labels for all the freq offsets slightly above the max value at that position
    for name, ppm in freqs.items():
        # Get the y-value at the specific ppm position
        ppm_index = np.argmin(np.abs(ppm_axis - ppm))
        max_val = np.max(data[:, ppm_index])
        # Only center 4.68 label, others aligned to left
        if ppm == 4.68:
            ha = "center"
        else:
            ha = "left"

        ax.text(
            ppm,
            max_val + label_offset,  # Slightly above the data line
            name,
            rotation=0,
            verticalalignment="top",
            horizontalalignment=ha,
            color="black",
            fontweight="bold",
        )
    return ax


def plot_spectra_over_time(
    df: pd.DataFrame,
    labels: list[str] | None = None,
    title: str = "MRS Unlocalized Spectra Over Time",
    **kwargs,
) -> sns.FacetGrid:
    """Plot MRS unlocalized spectra over time using a FacetGrid.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the spectra data with columns
        `["ppm", "Spectra_ID", "Intensity"]`.
    labels : list of str, optional
        Labels for the spectra. Defaults to an empty list.
    title : str, optional
        Figure title. Defaults to "MRS Unlocalized Spectra Over Time".

    Returns
    -------
    seaborn.FacetGrid
        The resulting FacetGrid.
    """
    if labels is None:
        labels = []

    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

    # 1. Initialize the FacetGrid
    pal = sns.cubehelix_palette(df["Spectra_ID"].nunique(), rot=-0.25, light=0.7)
    spectra_grid = sns.FacetGrid(
        df,
        row="Spectra_ID",  # Maps each spectrum to a column (facet)
        hue="Spectra_ID",
        sharey=True,  # Set to True if all spectra have a similar intensity range
        height=1,  # Height of each facet
        aspect=5,  # Aspect ratio of each facet
        palette=pal,
    )

    # 2. Map the plotting function
    spectra_grid.map(
        sns.lineplot,
        "ppm",  # Your X-axis
        "Intensity",  # Your Y-axis
    )

    # Add labels to each subplot
    if labels:
        for ax, label in zip(spectra_grid.axes.flatten(), labels):
            # Check how many lines the label will take
            n_lines = label.count("\n") + 1
            if n_lines > 1:
                ypos = 0.05 + (n_lines - 1) * 0.15
            else:
                ypos = 0.05

            ax.text(
                0.0,
                ypos,
                label,
                fontweight="bold",
                color=ax.lines[0].get_color(),
                ha="left",
                va="center",
                transform=ax.transAxes,
            )

    # 3. Apply standard spectral convention (reverse x-axis)
    spectra_grid.set(**_spectra_ax_params())  # Apply default spectra plotting params
    # Set x limits from 10 to 0 ppm
    spectra_grid.set(xlim=(10, -2))
    spectra_grid.set(**kwargs)

    spectra_grid.figure.subplots_adjust(hspace=-0.75)  # Overlay plots vertically

    # Remove axes details that don't play well with overlap
    spectra_grid.set_titles("")
    spectra_grid.set(yticks=[], ylabel="")
    spectra_grid.despine(bottom=True, left=True)
    spectra_grid.axes[-1][0].spines["bottom"].set_visible(True)

    # 4. Final adjustments
    spectra_grid.figure.suptitle(
        f"{title}",
        y=1.0,
        fontsize=12,
    )
    return spectra_grid
