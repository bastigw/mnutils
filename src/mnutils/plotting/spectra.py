from typing import Literal

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import nmrglue as ng
import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns
from IPython.display import display
from ipywidgets import FloatSlider, HBox, IntSlider, VBox, interactive
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from .. import utils

DEFAULT_SPECTRA_AX_PARAMS = {
    "xlim": (8.1, -2.1),
    "xlabel": "$^2$H chemical shift [ppm]",
    "ylabel": "signal [a.u.]",
}


class DefaultTickerParams:
    yticker_bins: int | Literal["auto"] = "auto"
    xticker_bins: int | Literal["auto"] = "auto"
    ticker_steps: list[int] = [1, 2, 5]


def set_default_ticks(ax: Axes) -> None:
    """Set default tick parameters for a given Axes object."""
    ax.yaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=DefaultTickerParams.yticker_bins,
            steps=DefaultTickerParams.ticker_steps,
        )
    )
    ax.xaxis.set_major_locator(
        ticker.MaxNLocator(
            nbins=DefaultTickerParams.xticker_bins,
            steps=DefaultTickerParams.ticker_steps,
        )
    )
    # By default override stylesheets behaviour of hiding ticks and set bottom and left to true
    ax.tick_params(bottom=True, left=True)


def set_default_spectra_ax_params(ax: Axes, **kwargs) -> None:
    """Set default spectra axis parameters for a given Axes object."""
    spectra_params = DEFAULT_SPECTRA_AX_PARAMS.copy()
    spectra_params.update(kwargs)
    ax.set(**spectra_params)


def plot_spectra(
    ppm: npt.NDArray,
    spectra: npt.NDArray,
    labels: list[str] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    if labels is None:
        labels = []
    fig = plt.figure(figsize=(7, 4))
    ax = plt.gca()
    plot_spectra_on_ax(ax, ppm, spectra, labels, **kwargs)
    return fig, ax


def plot_spectra_on_ax(
    ax: Axes,
    ppm: npt.NDArray[np.floating],
    spectra: npt.NDArray[np.complexfloating | np.floating],
    labels: list[str] | None = None,
    line_kwargs: dict | list[dict] | None = None,
    autophase: bool = False,
    p0: float | None = None,
    p1: float | None = None,
    **kwargs,
) -> list[Line2D]:
    if labels is None:
        labels = []

    # Initialize a list to store the plotted lines
    lines = []

    # To make sure we are working with a copy of the spectra
    spectra = np.copy(spectra)
    if spectra.ndim == 1:
        spectra = spectra[np.newaxis, :]  # Convert to 2D for consistency

    # Apply autophase correction or manual phasing with p0 and p1 for complex data
    if np.issubdtype(spectra.dtype, np.complexfloating):
        phased_spectra = utils.spectra.phase_nmr_data(
            spectra,  # type: ignore
            zero_order=p0,
            first_order=p1,
            autophase=autophase,
        )
    else:
        phased_spectra = spectra.astype(np.float64)

    if isinstance(line_kwargs, list):
        if len(line_kwargs) != (phased_spectra.shape[0]):
            raise ValueError(
                "Length of line_kwargs list must match number of phased_spectra."
            )
    elif isinstance(line_kwargs, dict):
        line_kwargs = [line_kwargs] * (phased_spectra.shape[0])

    for spec_idx in range(phased_spectra.shape[0]):
        if line_kwargs is None:
            kwargs_to_use = {}
        else:
            kwargs_to_use = line_kwargs[spec_idx]
        label = (
            labels[spec_idx] if spec_idx < len(labels) else f"Spectrum {spec_idx + 1}"
        )
        (line,) = ax.plot(
            ppm,
            phased_spectra[spec_idx],
            label=label,
            **kwargs_to_use,
        )
        lines.append(line)

    # Combine default spectra params with any provided kwargs
    set_default_ticks(ax)
    set_default_spectra_ax_params(ax, **kwargs)

    if phased_spectra.shape[0] > 1 or labels:
        ax.legend()

    # Return the list of plotted lines
    return lines


def plot_fid(
    fids: npt.NDArray[np.complexfloating | np.floating],
    header: dict | None = None,
    dwelltime: float | None = None,  # In seconds
    deadtime: float | None = None,  # In seconds
    labels: list[str] | None = None,
    **kwargs,
) -> tuple[Figure, Axes]:
    if labels is None:
        labels = []

    fig = plt.figure(figsize=(7, 6))
    ax = plt.gca()
    return fig, plot_fid_on_ax(
        ax,
        fids,
        header=header,
        dwelltime=dwelltime,
        deadtime=deadtime,
        labels=labels,
        **kwargs,
    )


def plot_fid_on_ax(
    ax: Axes,
    fids: npt.NDArray[np.complexfloating | np.floating],
    header: dict | None = None,
    dwelltime: float | None = None,  # In seconds
    deadtime: float | None = None,  # In seconds
    labels: list[str] | None = None,
    show_real: bool = True,
    show_imag: bool = True,
    **kwargs,
) -> Axes:
    if labels is None:
        labels = []

    # Make sure we are working with a copy of the fids
    fids = np.copy(fids)

    npts = fids.shape[-1]
    xaxis = np.arange(npts)
    xaxis_label = "Points"
    # If header in kwargs, extract dwell time and create time axis in seconds
    if header is not None:
        try:
            dwelltime = 1.0 / header["rdb_hdr"]["spectral_width"]
            deadtime = header["rdb_hdr"]["te"] * 1e-6  # Convert from us to s
            xaxis = (np.arange(0, npts) * dwelltime + deadtime) * 1e3  # in ms
            xaxis_label = "Time [ms]"
        except KeyError:
            xaxis = np.arange(npts)
    elif dwelltime is not None and deadtime is not None:
        xaxis = (np.arange(0, npts) * dwelltime + deadtime) * 1e3  # in ms
        xaxis_label = "Time [ms]"

    # Handle case where fids is 1D
    if fids.ndim == 1:
        fids = fids[np.newaxis, :]  # Convert to 2D for consistency

    for i in range(fids.shape[0]):
        label_real = f"{labels[i]} (Real)" if i < len(labels) else f"FID {i + 1} (Real)"
        label_imag = f"{labels[i]} (Imag)" if i < len(labels) else f"FID {i + 1} (Imag)"
        if show_real:
            ax.plot(xaxis, np.real(fids[i, :]), label=label_real)
        # Only show imaginary part if it has one
        if show_imag and np.iscomplexobj(fids):
            ax.plot(xaxis, np.imag(fids[i, :]), label=label_imag, linestyle="--")

    ax.legend()
    ax.set(**kwargs)

    # Set default FID axis parameters
    ax.set_xlabel(xaxis_label)
    ax.set_ylabel("FID signal [a.u.]")

    # Set default ticks
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


def create_interactive_phase_correction(
    ppm_axis: npt.NDArray[np.float64],
    spectrum: npt.NDArray[np.complex64],
    do_initial_autophase: bool = True,
    plot_magnitude_spectrum: bool = True,
    **kwargs,
) -> tuple[FloatSlider | IntSlider, FloatSlider | IntSlider]:
    # Create the figure and axis once
    fig, ax = plt.subplots(figsize=(10, 7))
    (line,) = ax.plot([], [], label="Phased Data")  # Initialize an empty line
    if plot_magnitude_spectrum:
        ax.plot(
            ppm_axis,
            np.abs(spectrum),
            label="Magnitude Spectrum",
            color="gray",
            alpha=0.5,
        )

    default_phase_correction_0 = 0.0
    default_phase_correction_1 = 0.0
    if do_initial_autophase:
        _, opt = ng.proc_autophase.autops(
            spectrum,
            p0=default_phase_correction_0,
            p1=default_phase_correction_1,
            fn="acme",
            disp=0,
            return_phases=True,
        )
        # Estimate initial phase correction values
        default_phase_correction_0 = opt[0]
        default_phase_correction_1 = opt[1]

    # Get index of ppm axis closest to 4.68 ppm for pivot default
    pivot_default = np.argmin(np.abs(ppm_axis - 4.68))
    pivotLine = ax.axvline(
        ppm_axis[pivot_default], color="r", alpha=0.5
    )  # Add a vertical line for pivot
    set_default_ticks(ax)
    set_default_spectra_ax_params(ax, **kwargs)
    ax.legend()

    def phasecorr(phase_correction_0, phase_correction_1, pivot, xlim_start, xlim_end):
        # Calculate the phased data
        phased_data = ng.proc_base.ps(spectrum, phase_correction_0, phase_correction_1)

        # Update the line data
        line.set_data(ppm_axis, np.real(phased_data))
        pivotLine.set_xdata(
            [ppm_axis[pivot], ppm_axis[pivot]]
        )  # Update pivot line position
        ax.set_xlim(xlim_start, xlim_end)
        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

        # Calculate and print phase correction values
        # Update the title with phase correction values
        ax.set_title(
            f"Phase Correction: p0 = {phase_correction_0:.2f}, p1 = {phase_correction_1:.2f}"
        )

    widgets = interactive(
        phasecorr,
        phase_correction_0=FloatSlider(
            value=default_phase_correction_0,
            min=-180,
            max=180,
            step=0.1,
            description="Phase 0 (deg):",
        ),
        phase_correction_1=FloatSlider(
            value=default_phase_correction_1,
            min=-180,
            max=180,
            step=0.1,
            description="Phase 1 (deg):",
        ),
        pivot=IntSlider(
            value=pivot_default,
            min=0,
            max=spectrum.size,
            step=1,
            description="Pivot Index:",
        ),
        xlim_start=FloatSlider(
            value=10,
            min=ppm_axis.min(),
            max=ppm_axis.max(),
            step=0.5,
            description="X-axis start (ppm):",
        ),
        xlim_end=FloatSlider(
            value=-2,
            min=ppm_axis.min(),
            max=ppm_axis.max(),
            step=0.5,
            description="X-axis end (ppm):",
        ),
    )

    phaseVBox = VBox(widgets.children[:3])
    controls = HBox([phaseVBox, widgets.children[3], widgets.children[4]])
    display(controls)

    return widgets.children[:2]


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
    spectra_grid.set(
        **DEFAULT_SPECTRA_AX_PARAMS
    )  # Apply default spectra plotting params
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
