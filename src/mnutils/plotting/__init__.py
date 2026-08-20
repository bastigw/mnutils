from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..utils import file_helpers
from . import images, spectra  # noqa: F401

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Helvetica", "Liberation Sans", "Arial", "sans-serif"]
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.constrained_layout.use"] = True

contexts = ["poster", "talk", "paper"]

DEFAULT_SAVE_PARAMS = {
    "dpi": 150,
    "format": "png",
    "transparent": False,
    "bbox_inches": "tight",
}

POSTER_SAVE_PARAMS = DEFAULT_SAVE_PARAMS.copy()
POSTER_SAVE_PARAMS.update(
    {
        "dpi": 300,
    }
)

TALK_SAVE_PARAMS = DEFAULT_SAVE_PARAMS.copy()
TALK_SAVE_PARAMS.update(
    {
        "transparent": True,
    }
)

PAPER_SAVE_PARAMS = DEFAULT_SAVE_PARAMS.copy()
PAPER_SAVE_PARAMS.update(
    {
        "dpi": 300,
        "pad_inches": 0.1,
    }
)


def save_figure(
    fig: Figure,
    folder: str | Path,
    filename: str,
    prepend_date: bool = True,
    move_old: bool = True,
    context: Literal["poster", "talk", "paper"] | None = None,
    modify_folder_path: bool = True,
    **kwargs,
) -> None:
    """Save a matplotlib figure to a file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    folder : str or Path
        The folder to save the figure in.
    filename : str
        The name of the file to save the figure as (without extension).
    prepend_date : bool, optional
        Whether to prepend the current date and time to the filename.
        Defaults to True.
    move_old : bool, optional
        Whether to move old files with the same name to an "old" folder
        first. Defaults to True.
    context : {"poster", "talk", "paper"}, optional
        Preset save parameters (dpi, transparency, padding) to use.
    modify_folder_path : bool, optional
        If True and `context` is given, append `context` to `folder` when
        it isn't already the folder's last component. Defaults to True.
    **kwargs
        Additional keyword arguments passed to `fig.savefig`.
    """
    folder = Path(folder)
    # If folder does not contain "context" as last directory append it to the folder path
    if modify_folder_path and context is not None and folder.name != context:
        folder = folder / context

    # Combine default save parameters with any provided kwargs
    match context:
        case "poster":
            save_params = POSTER_SAVE_PARAMS.copy()
        case "talk":
            save_params = TALK_SAVE_PARAMS.copy()
        case "paper":
            save_params = PAPER_SAVE_PARAMS.copy()
        case None:
            save_params = DEFAULT_SAVE_PARAMS.copy()
    save_params.update(kwargs)

    # Create folder if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)
    if move_old:
        file_helpers.move_files_with_glob(folder, f"*{filename}.*")

    if prepend_date:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_str}_{filename}"
    file_extension = save_params.get("format", DEFAULT_SAVE_PARAMS["format"])
    full_path = folder / f"{filename}.{file_extension}"

    fig.savefig(full_path, **save_params)


def save_current_figure(
    folder: str | Path, filename: str, prepend_date: bool = True, **kwargs
) -> None:
    """Save the current matplotlib figure to a file.

    Parameters
    ----------
    folder : str or Path
        The folder to save the figure in.
    filename : str
        The name of the file to save the figure as (without extension).
    prepend_date : bool, optional
        Whether to prepend the current date and time to the filename.
        Defaults to True.
    **kwargs
        Additional keyword arguments forwarded to `save_figure`.
    """
    fig = plt.gcf()
    save_figure(fig, folder, filename, prepend_date, **kwargs)
