from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyAMARES
import xarray as xr
import xmris  # noqa: F401
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..utils import file_helpers

DEFAULT_INIT_PARAMS = {
    "preview": False,
    "xlim": (5, -5),
    "g_global": 0.8,
    "ppm_offset": -4.68,  # For now default to water offset
    "normalize_fid": False,
}

DEFAULT_FIT_PARAMS = {
    "method": "leastsq",
    "ifplot": False,
    "inplace": False,
}

DEFAULT_BATCH_FIT_PARAMS = {
    "num_workers": 10,
    "initialize_with_lm": True,
    "method": "leastsq",
}

DEFAULT_PLOT_PARAMS = {"ifphase": True}


def fit_single_fid(
    fid: npt.NDArray | xr.DataArray,
    init_params: dict,
    fit_params: dict | None = None,
    raw_header: dict | None = None,
    filter_by_ppm: tuple[float, float] | None = None,
):
    """Fit a single FID using the pyAMARES fitting library.

    Parameters
    ----------
    fid : ndarray or xarray.DataArray
        The input FID (time-domain signal) to be fitted.
    init_params : dict
        Initial parameters for fitting. Must include `priorknowledgefile`,
        `MHz`, `sw`, and `deadtime` (directly or via `raw_header`).
    fit_params : dict, optional
        Fitting parameters, merged over `DEFAULT_FIT_PARAMS`.
    raw_header : dict, optional
        Raw scanner header, used to derive `MHz`/`sw`/`deadtime` defaults.
    filter_by_ppm : tuple[float, float], optional
        If given, restrict fitting to this ppm range.

    Returns
    -------
    pyAMARES fit result
        The fitted FID result object.

    Raises
    ------
    ValueError
        If a required field (`priorknowledgefile`, `MHz`, `sw`, `deadtime`)
        is missing from the merged init parameters.
    """
    if fit_params is None:
        fit_params = {}

    # Merge default parameters with user-provided parameters
    init_kwargs = DEFAULT_INIT_PARAMS.copy()
    if raw_header is not None:
        init_kwargs.update(vars(pyAMARES.fileio.readfidall.header2par_v73(raw_header)))
    # Then override with any provided init_params
    init_kwargs.update(init_params)

    # Init kwargs must have the fields priorknowledgefile, MHz, sw, and deadtime
    # Check for those fields and raise error if missing
    required_fields = ["priorknowledgefile", "MHz", "sw", "deadtime"]
    for field in required_fields:
        if field not in init_kwargs:
            raise ValueError(f"Missing required field '{field}' in init_params.")

    logger.debug(f"Initializing FID with parameters: {init_kwargs}")

    if isinstance(fid, xr.DataArray):
        fid = fid.values.squeeze()

    FIDobj = pyAMARES.initialize_FID(
        fid,
        **init_kwargs,
    )

    fit_kwargs = DEFAULT_FIT_PARAMS.copy()
    # Default to plot for single voxel
    fit_kwargs["ifplot"] = True
    fit_kwargs.update(fit_params)
    # By default turn of linebroadening during plot
    FIDobj.plotParameters.lb = 0
    fit_kwargs["fid_parameters"] = FIDobj
    fit_kwargs["fitting_parameters"] = FIDobj.initialParams

    if filter_by_ppm is not None:
        logger.info(f"Filtering FID data by ppm range: {filter_by_ppm}")
        FIDobj = pyAMARES.filter_fid_by_ppm(FIDobj, fit_ppm=filter_by_ppm, ifplot=True)
        filtered_params = pyAMARES.filter_param_by_ppm(
            allpara=FIDobj.initialParams, fit_ppm=filter_by_ppm, MHz=FIDobj.MHz
        )
        fit_kwargs["fid_parameters"] = FIDobj
        fit_kwargs["fitting_parameters"] = filtered_params

    logger.debug(f"Fitting FID with parameters: {fit_kwargs}")

    return pyAMARES.fitAMARES(**fit_kwargs)


def fit_multiple_fids(
    fids: npt.NDArray | xr.DataArray,
    batch_fitting_params: dict | None = None,
    init_params: dict | None = None,
    batch_fitting_obj=None,
    raw_header: dict | None = None,
) -> list:
    """Batch-fit multiple FIDs in parallel using the pyAMARES library.

    Initializes a batch fitting object from the average FID of the highest
    SNR voxels, then applies parallel fitting to every input FID.

    Parameters
    ----------
    fids : ndarray or xarray.DataArray
        The input FIDs (time-domain signals) to be fitted. Expected shape is
        `(n_signals, n_points)`.
    batch_fitting_params : dict, optional
        Parameters for batch fitting, merged over `DEFAULT_BATCH_FIT_PARAMS`.
    init_params : dict, optional
        Initial parameters for fitting, merged over `DEFAULT_INIT_PARAMS`.
    batch_fitting_obj : optional
        A pre-initialized batch fitting object to reuse instead of creating
        a new one.
    raw_header : dict, optional
        Raw header information passed to the initial single-FID fit.

    Returns
    -------
    list[tuple]
        The fitting results for each FID, each including the fitted
        parameters and other relevant information.

    Notes
    -----
    The batch fitting object is initialized using the average FID of the 10
    highest-SNR voxels (or fewer if there are fewer than 10 FIDs).
    `pyAMARES.run_parallel_fitting_with_progress` performs the parallel
    fitting with a progress indicator.
    """
    if batch_fitting_params is None:
        batch_fitting_params = {}
    if init_params is None:
        init_params = {}

    batch_fit_kwargs = DEFAULT_BATCH_FIT_PARAMS.copy()
    batch_fit_kwargs.update(batch_fitting_params)

    if isinstance(fids, xr.DataArray):
        fids = fids.transpose("voxel", "time").values

    if batch_fitting_obj is None:
        init_kwargs = DEFAULT_INIT_PARAMS.copy()
        init_kwargs.update(init_params)

        logger.debug(f"Initializing batch fitting with parameters: {init_kwargs}")

        # The input is multiple FIDs. To initialize the batch object take the average fid
        # of the 10 highest SNR voxels
        n_voxels = 10 if fids.shape[0] >= 10 else fids.shape[0]
        SNRs = np.apply_along_axis(pyAMARES.fidSNR, 1, fids)
        avg_fid = np.mean(fids[np.argsort(SNRs)[-n_voxels:]], axis=0)  # type: ignore

        fit_kwargs = {
            "ifplot": False,
        }
        fit_kwargs["method"] = batch_fit_kwargs.get("method", DEFAULT_BATCH_FIT_PARAMS["method"])

        batch_fitting_obj = fit_single_fid(
            avg_fid,
            init_params=init_kwargs,
            fit_params=fit_kwargs,
            raw_header=raw_header,
        )

    results = pyAMARES.run_parallel_fitting_with_progress(
        fids,
        FIDobj_shared=batch_fitting_obj,
        initial_params=batch_fitting_obj.fittedParams,  # pyright: ignore[reportAttributeAccessIssue]
        **batch_fit_kwargs,
    )

    logger.debug(
        f"Completed batch fitting of {len(results)} FIDs. Results type: {type(results)}, "
        f"first element type: {type(results[0]) if results else 'N/A'}"
    )

    # Print debug messages about the first result. It should be a tuple of
    # (DataFrame, MinimizerResult)
    if results:
        logger.debug(f"First result content: {results[0]}")

    return results


@dataclass
class FitResults:
    """Container for extracted pyAMARES batch fitting results."""

    badfit_ids: npt.NDArray
    goodness_of_fit: pd.DataFrame
    combined_fit_results: pd.DataFrame


def extract_from_fit_results(
    fit_results: list[tuple],
    bad_fit_chisqr_threshold: float = -1.0,
    ppm_offset: float = DEFAULT_INIT_PARAMS["ppm_offset"],
    index_key: str = "voxel_id",
    index_values: list | pd.Index | None = None,
) -> FitResults:
    """Extract goodness-of-fit metrics and combined fit parameters from batch fit results.

    Parameters
    ----------
    fit_results : list of tuple
        Per-voxel results as returned by `fit_multiple_fids`, each a
        `(DataFrame, MinimizerResult)` tuple.
    bad_fit_chisqr_threshold : float, optional
        Chi-squared threshold above which a fit is flagged as bad. If
        negative (default), it is set to 3x the median chi-squared.
    ppm_offset : float, optional
        Offset subtracted from the fitted chemical shift values. Defaults
        to `DEFAULT_INIT_PARAMS["ppm_offset"]`.
    index_key : str, optional
        Name used for the voxel index in the returned DataFrames. Defaults
        to "voxel_id".
    index_values : list or pandas.Index, optional
        Values to use for the voxel index. Must match the length of
        `fit_results` if provided. Defaults to a range index.

    Returns
    -------
    FitResults
        The bad fit ids, goodness-of-fit metrics, and combined fit results.
    """
    # Extract goodness of fit metrics
    GoF_params_to_extract = ["chisqr", "redchi"]
    goodness_of_fit_metrics = np.full(
        (len(fit_results), len(GoF_params_to_extract)), np.nan, dtype=np.float64
    )

    # Assert that if index_values is provided, its length matches fit_results
    if index_values is not None and len(index_values) != len(fit_results):
        raise ValueError("Length of index_values must match the number of fit_results.")

    if index_values is None:
        index_values = list(range(len(fit_results)))

    for idx, result in enumerate(fit_results):
        # The minimizer results are stored in result[1]
        # Extract chisqr, redchi, aic, and bic from it. AIC and BIC should not be valid for
        # this analysis where we get goodness of fit
        if isinstance(result, tuple) and len(result) > 1 and result[1] is not None:
            minimizer_result = result[1]
            for i, param in enumerate(GoF_params_to_extract):
                if not hasattr(minimizer_result, param):
                    logger.warning(f"Minimizer result does not have attribute '{param}'")
                else:
                    goodness_of_fit_metrics[idx, i] = getattr(minimizer_result, param)
        else:
            logger.warning(
                f"Result at {idx} is not properly initialized or missing minimizer result."
            )

    gof_metrics_df = pd.DataFrame(
        goodness_of_fit_metrics,
        columns=GoF_params_to_extract,
        index=pd.Index(index_values, name=index_key),
    )

    # Select bad fits
    badfit_ids = []
    # Now check for bad fits based on chisqr
    if gof_metrics_df["chisqr"].isna().all():
        median_chisqr = np.nan
    else:
        median_chisqr = np.nanmedian(gof_metrics_df["chisqr"]).astype(float)
    if bad_fit_chisqr_threshold < 0:
        bad_fit_chisqr_threshold = 3 * median_chisqr

    badfit_ids = gof_metrics_df.index[
        (gof_metrics_df["chisqr"] >= bad_fit_chisqr_threshold) | gof_metrics_df.isna().any(axis=1)
    ].to_numpy()

    logger.trace(
        f"Bad fits identified: {len(badfit_ids)} out of {len(gof_metrics_df)}. "
        f"Median chisqr={median_chisqr:.2f}, Threshold={bad_fit_chisqr_threshold:.2f}"
    )

    # Finally combine the result table into a big multi-index DataFrame mapping to more
    # consistent naming
    # Prepare the results for concatenation
    # Fill in the missing results with NaN DataFrames of the same shape and structure
    # filled with np.nan
    # Select an index that is not in badfit_ids to get the structure
    if badfit_ids.size != 0:
        valid_index = next((i for i in range(len(fit_results)) if i not in badfit_ids), None)
        if valid_index is not None:
            template_df = fit_results[valid_index][0].T.copy()
            nan_df = pd.DataFrame(np.nan, index=template_df.index, columns=template_df.columns)
        else:
            logger.warning(
                "No valid index found to create a template DataFrame. This could mean all "
                "fits are bad. Or something else went wrong"
            )
            nan_df = pd.DataFrame()
    else:
        logger.trace("No bad fits detected; all results are valid.")
        nan_df = pd.DataFrame()

    # Replace missing or invalid results with NaN DataFrames
    prepared_results = [
        result[0].T if result is not None and result[0] is not None else nan_df.copy()
        for result in fit_results
    ]

    combined_fit_results_df = pd.concat(
        prepared_results,
        keys=index_values,
        names=[index_key, "parameter"],
    )

    combined_fit_results_df.reset_index(inplace=True, drop=False)
    mapping = {
        "CRLB(%)": ("amplitude", "crlb"),
        "CRLB(LW%)": ("lw", "crlb"),
        "CRLB(cs%) ": ("chem_shift", "crlb"),
        "CRLB(phase%)": ("phase", "crlb"),
        "LW(Hz)": ("lw", "value"),
        "SNR": ("snr", "value"),
        "amplitude": ("amplitude", "value"),
        "chem shift(ppm)": ("chem_shift", "value"),
        "g": ("g", "value"),
        "g (%)": ("g", "crlb"),
        "g_sd": ("g", "sd"),
        "phase(deg)": ("phase", "value"),
        "sd": ("amplitude", "sd"),
        "sd(Hz)": ("lw", "sd"),
        "sd(deg)": ("phase", "sd"),
        "sd(ppm)": ("chem_shift", "sd"),
    }
    # Based on the mapping add the value type and rename the parameter name
    combined_fit_results_df["value_type"] = combined_fit_results_df["parameter"].map(
        lambda x: mapping[str(x)][1]
    )
    combined_fit_results_df["parameter"] = combined_fit_results_df["parameter"].map(
        lambda x: mapping[str(x)][0]
    )
    combined_fit_results_df.set_index([index_key, "parameter", "value_type"], inplace=True)
    # Adjust chem_shift values by ppm_offset (subtract as ppm_offset is defined as water - 4.68)
    combined_fit_results_df.loc[(slice(None), "chem_shift", "value"), :] -= ppm_offset

    return FitResults(
        badfit_ids=badfit_ids,
        goodness_of_fit=gof_metrics_df,
        combined_fit_results=combined_fit_results_df,
    )


def save_extracted_results(
    path: str | Path,
    fit_results: FitResults,
    attributes_to_add: dict[str, str | int] | None = None,
    additional_datasets: dict[str, npt.NDArray] | None = None,
    prepend_datetime: bool = False,
    move_old: bool = True,
) -> Path:
    """Save the extracted fitting results to an HDF5 file.

    Parameters
    ----------
    path : str or Path
        The file path where the results will be saved. Forced to a `.h5`
        suffix if it doesn't already end in `.h5`/`.hdf5`.
    fit_results : FitResults
        The fitting results to save (badfit IDs, goodness-of-fit metrics,
        and combined fit results).
    attributes_to_add : dict[str, str or int], optional
        Additional attributes to add to the HDF5 file.
    additional_datasets : dict[str, ndarray], optional
        Additional datasets to write into the HDF5 file.
    prepend_datetime : bool, optional
        Whether to prepend the current datetime to the filename.
    move_old : bool, optional
        Whether to move any existing file with the same stem into an "old"
        subfolder first. Defaults to True.

    Returns
    -------
    Path
        The path to the saved file.

    Raises
    ------
    FileExistsError
        If the target path already exists (after any datetime prefix).
    OSError, KeyError, ValueError
        If writing the HDF5 file fails.
    """
    if attributes_to_add is None:
        attributes_to_add = {}
    if additional_datasets is None:
        additional_datasets = {}

    try:
        # Check if path parent directory exists, if not create it
        path = Path(path)
        if not path.parent.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
        # Check if filename ends with .h5 or .hdf5
        if path.suffix not in [".h5", ".hdf5"]:
            # Change suffix to .h5
            path = path.with_suffix(".h5")

        if move_old:
            file_helpers.move_files_with_glob(path.parent, f"*{path.stem}.*")

        # Prepend datetime to filename if requested
        if prepend_datetime:
            current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.with_name(f"{current_datetime}_{path.name}")

        # Check if file exists and log an error and raise exception
        if path.exists():
            logger.error(f"File {path} already exists. Aborting to prevent overwrite.")
            raise FileExistsError(f"File {path} already exists.")

        # Add default attribute SAVED_ON with current datetime if not provided
        if "SAVED_ON" not in attributes_to_add:
            attributes_to_add["SAVED_ON"] = datetime.now().isoformat()

        with pd.HDFStore(path, mode="w") as store:
            store.put(
                "goodness_of_fit",
                fit_results.goodness_of_fit,
                format="table",
                data_columns=True,
            )
            store.put(
                "combined_fit_results",
                fit_results.combined_fit_results,
                format="table",
                data_columns=True,
            )
            # Save badfit_ids as a Series
            store.put(
                "badfit_ids",
                pd.Series(fit_results.badfit_ids),
                data_columns=True,
            )

        # Add additional attributes and datasets to the HDF5 file in a single open-close operation
        with h5py.File(path, "r+") as h5file:
            for key, value in attributes_to_add.items():
                h5file.attrs[key] = value

            if additional_datasets:
                for dataset_name, dataset_value in additional_datasets.items():
                    if dataset_name in h5file:
                        logger.warning(
                            f"Dataset '{dataset_name}' already exists and will be overwritten."
                        )
                        del h5file[dataset_name]
                    h5file.create_dataset(
                        dataset_name, data=dataset_value, dtype=dataset_value.dtype
                    )

    except (OSError, KeyError, ValueError) as e:
        logger.exception(f"Failed to save extracted results to {path}: {e}")
        raise
    return Path(path)


def load_extracted_results(path: str | Path) -> tuple[FitResults, dict]:
    """Load extracted fitting results from an HDF5 file.

    Parameters
    ----------
    path : str or Path
        The file path from which the results will be loaded.

    Returns
    -------
    tuple[FitResults, dict]
        The loaded fitting results, and a dictionary of user attributes
        stored alongside them.

    Raises
    ------
    FileNotFoundError, KeyError, OSError, ValueError
        If reading the HDF5 file fails.
    """
    try:
        with pd.HDFStore(path, mode="r") as store:
            goodness_of_fit = pd.DataFrame(store["goodness_of_fit"])
            combined_fit_results = pd.DataFrame(store["combined_fit_results"])
            badfit_ids = store["badfit_ids"].to_numpy()
            # Get all additionally stored attributes
            attrs = store.root._v_attrs
            user_attributes = {name: getattr(attrs, name) for name in attrs._v_attrnamesuser}

        fit_results = FitResults(
            badfit_ids=badfit_ids,
            goodness_of_fit=goodness_of_fit,
            combined_fit_results=combined_fit_results,
        )
        return fit_results, user_attributes
    except (FileNotFoundError, KeyError, OSError, ValueError) as e:
        logger.exception(f"Failed to load extracted results from {path}: {e}")
        raise


def plot_amares_fitting(
    result,
    title: str | None = None,
    fig_kwargs: dict | None = None,
    ax_kwargs: dict | None = None,
    autophase: bool = DEFAULT_PLOT_PARAMS["ifphase"],
    add_table: bool = False,
) -> Figure:
    """Plot an AMARES fitting result, optionally with a parameter table.

    Parameters
    ----------
    result : pyAMARES fit result
        The result object containing the fitting parameters and data.
    title : str, optional
        The title of the plot. Defaults to "AMARES Fitting Result".
    fig_kwargs : dict, optional
        Additional keyword arguments for `plt.subplots`.
    ax_kwargs : dict, optional
        Additional keyword arguments applied to the main axes via `ax.set`.
    autophase : bool, optional
        Whether to apply autophase to the data. Defaults to
        `DEFAULT_PLOT_PARAMS["ifphase"]`.
    add_table : bool, optional
        Whether to add a parameter table beneath the plot. Defaults to False.

    Returns
    -------
    matplotlib.figure.Figure
        The combined figure.
    """
    if fig_kwargs is None:
        fig_kwargs = {}
    if ax_kwargs is None:
        ax_kwargs = {}

    fitted_params = result.fittedParams
    plotParameters = result.plotParameters
    ppm = result.ppm - result.ppm_offset
    xlim = np.subtract(plotParameters.xlim, result.ppm_offset)
    xlabel = "ppm"
    mode = "real"
    label = "Fitted Spectrum"
    p_pd = result.amares_to_plot_pd
    # Offset p_pd by ppm_offset
    p_pd["freq"] = p_pd["freq"] - result.ppm_offset
    fid = result.fid
    fid_fit = pyAMARES.kernel.fid.fft_params(
        timeaxis=result.timeaxis, params=fitted_params, fid=True
    )
    hsvdarr = pyAMARES.kernel.fid.fft_params(
        result.timeaxis, fitted_params, fid=True, return_mat=True
    ).T
    if autophase:
        from nmrglue.process import proc_autophase, proc_base

        fid, opt = proc_autophase.autops(fid, "acme", return_phases=True, disp=False)
        # Then apply the same phase to the fitted fid
        fid_fit = proc_base.ps(fid_fit, p0=opt[0], p1=opt[1])
        # And apply to hsvdarr
        hsvdarr = proc_base.ps(hsvdarr, p0=opt[0], p1=opt[1])
        # Turn of future phasing in plotParameters
        plotParameters.ifphase = False

    if not add_table:
        # Plot with single axis
        fig, ax_combined = plt.subplots(figsize=(10, 6), **fig_kwargs)
    else:
        # Plot with two axes to accommodate table
        fig, axes = plt.subplots(2, 1, height_ratios=[2, 1], figsize=(10, 8), **fig_kwargs)
        ax_combined: Axes = axes[0]
        ax_table: Axes = axes[1]
        ax_table.axis("off")

        try:
            # Get table by extracting from results
            extracted_results = extract_from_fit_results(
                [(result.result_multiplets, result.out_obj)]
            )
            table = extracted_results.combined_fit_results.loc[0].T
            # Check if table is dataframe else raise error
            if not isinstance(table, pd.DataFrame):
                raise ValueError(
                    "Extracted table is not a pandas DataFrame. Something went wrong "
                    "during results extraction."
                )
            add_colored_table_to_plot(ax_table, table)
            # Below the table add a text with the chisqr and redchi values if the
            # result.out_obj is valid and has those attributes
            if result.out_obj is not None:
                chisqr = getattr(result.out_obj, "chisqr", np.nan)
                redchi = getattr(result.out_obj, "redchi", np.nan)
                if chisqr is not np.nan or redchi is not np.nan:
                    ax_table.text(
                        0.5,
                        1,
                        f"Fit Metrics: Chi-squared: {chisqr:.5g}, "
                        f"Reduced Chi-squared: {redchi:.5g}",
                        ha="center",
                        va="top",
                        fontsize=10,
                    )

        except (KeyError, ValueError, TypeError, AttributeError) as e:
            logger.exception(f"Could not add table to plot: {e}")

    pyAMARES.util.visualization.preview_HSVD(ax_combined, hsvdarr, ppm, p_pd, xlim)
    # Remove latest line plotted on ax_combined (the hsvdarr summed spectrum)
    ax_combined.lines[-1].remove()
    pyAMARES.util.visualization.plot_fit(
        ax_combined, fid, fid_fit, ppm, xlim, mode, label, plotParameters=plotParameters
    )
    # Set x limits
    ax_combined.set_xlabel(xlabel)
    ax_combined.set_xlim(xlim)
    ax_combined.set(**ax_kwargs)
    fig.suptitle(title if title else "AMARES Fitting Result")

    return fig


def add_colored_table_to_plot(table_ax: Axes, table: pd.DataFrame, sig: int = 3) -> Axes:
    """Add a colored table beneath the given axis in the plot.

    Parameters
    ----------
    table_ax : matplotlib.axes.Axes
        The axis to which the table will be added.
    table : pandas.DataFrame
        The table data to be displayed, with a 2-level MultiIndex column
        (parameter, value_type).
    sig : int, optional
        The number of significant digits to round the table values to.
        Defaults to 3.

    Returns
    -------
    matplotlib.axes.Axes
        The axis the table was added to.
    """
    from matplotlib.table import Cell as mplCell

    rounded_table = table.map(lambda x: f"{x:5.{sig}g}".strip())

    # Prepare MultiIndex column labels as strings for display
    col_labels = [f"{col[1]}" for col in rounded_table.columns.values]
    row_labels = rounded_table.index.tolist()

    # Define colors for each value type
    value_type_colors = {
        "value": "#e6f2ff",
        "sd": "#fff2cc",
        "crlb": "#f9d5e5",
    }
    # Map each column to its value type
    col_value_types = [col[1] for col in rounded_table.columns]

    n_rows, n_cols = rounded_table.shape

    tbl = table_ax.table(
        cellText=rounded_table.to_numpy().tolist(),
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc="center",
        loc="center",
    )

    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.2)

    # Style header
    for col_idx in range(n_cols):
        cell = tbl[(0, col_idx)]
        cell.set_fontsize(12)
        cell.set_text_props(weight="bold", color="white")
        cell.set_facecolor("#40466e")

    # Style row labels
    for row_idx in range(n_rows):
        cell = tbl[(row_idx + 1, -1)]
        cell.set_fontsize(12)
        cell.set_text_props(weight="bold", color="#40466e")
        cell.set_facecolor("#e8eaf6")

    # Style data cells by value type
    for col_idx, value_type in enumerate(col_value_types):
        for row_idx in range(n_rows):
            cell = tbl[(row_idx + 1, col_idx)]
            cell.set_facecolor(value_type_colors.get(value_type, "white"))

    # Add a row of cells above the header by getting the cells and adding a custom
    # rectangle above the header that spans three columns
    # First figure out how often the multi-index header appears
    column_names = table.columns.get_level_values(0).to_list()
    from itertools import groupby

    # This gives a list of tuples with (column name, count)
    cells = tbl.get_celld()
    height = cells[1, 0].get_height()
    cell_width = cells[1, 0].get_width()
    columns_counted = [(k, len(list(g))) for k, g in groupby(column_names)]
    start_col = 0

    # IMPORTANT to update the positions of the cells before querying their positions
    table_ax.figure.canvas.draw()
    for col_name, count in columns_counted:
        width = cell_width * count
        xy = cells[0, start_col].get_xy()
        # Move the new cell up by its height
        xy = (xy[0], xy[1] + height)
        new_cell = mplCell(xy, width=width, height=height, text=col_name, loc="center")
        table_ax.add_patch(new_cell)
        start_col += count

    return table_ax


def priorknowledge_to_multiindex(
    pk_df: pd.DataFrame, sections: list[str] | None = None
) -> pd.DataFrame:
    """Convert a prior-knowledge DataFrame into a MultiIndex DataFrame.

    Parameters
    ----------
    pk_df : pandas.DataFrame
        The prior-knowledge DataFrame.
    sections : list of str, optional
        Section headers to group by. Defaults to
        `["Initial Values", "Bounds"]`.

    Returns
    -------
    pandas.DataFrame
        The DataFrame reindexed with a `(Group, Index)` MultiIndex.
    """
    if sections is None:
        sections = ["Initial Values", "Bounds"]

    # Reset index to ensure 'Index' is a column
    if pk_df.index.name == "Index":
        pk_df = pk_df.reset_index()

    mask = pk_df["Index"].isin(sections)
    # Create a new column 'Group' and populate it only at the header rows
    pk_df["Group"] = None
    pk_df.loc[mask, "Group"] = pk_df.loc[mask, "Index"]
    # Use forward fill to propagate the Group name to the rows below
    pk_df["Group"] = pk_df["Group"].ffill()
    # Remove the rows that were just header markers
    pk_df = pk_df[~mask]
    pk_df.set_index(["Group", "Index"], inplace=True)
    return pk_df


def multiindex_to_priorknowledge(
    pk_df: pd.DataFrame, group_order: list[str] | None = None
) -> pd.DataFrame:
    """Convert a MultiIndex prior-knowledge DataFrame back into a raw DataFrame.

    Parameters
    ----------
    pk_df : pandas.DataFrame
        The MultiIndex prior-knowledge DataFrame.
    group_order : list of str, optional
        Section headers to group and sort by. Defaults to
        `["Initial Values", "Bounds"]`.

    Returns
    -------
    pandas.DataFrame
        The flattened, sorted DataFrame.
    """
    if group_order is None:
        group_order = ["Initial Values", "Bounds"]

    # Reset the index to flatten the MultiIndex
    raw_df = pk_df.reset_index()
    # Extract the 'Group' column and set it as a new row where the group changes
    group_rows = raw_df["Group"].drop_duplicates().reset_index(drop=True)
    group_rows = pd.DataFrame({"Index": group_rows, "Group": None})
    # Append the group rows back to the dataframe
    raw_df = pd.concat([group_rows, raw_df], ignore_index=True)
    # Fill the 'Group' column with the 'Index' values where applicable
    raw_df["Group"] = raw_df["Group"].fillna(raw_df["Index"])
    # Sort the dataframe by the "Group" column, ensuring "Initial Values" comes before "Bounds"
    raw_df["Group"] = pd.Categorical(raw_df["Group"], categories=group_order, ordered=True)
    raw_df.sort_values(by="Group", inplace=True)
    # Reset the index
    raw_df.set_index("Index", inplace=True)
    # Drop the 'Group' column and set 'Index' as the index
    raw_df.drop(columns=["Group"], inplace=True)
    return raw_df
