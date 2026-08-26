"""Saving matplotlib figures with per-context (poster/talk/paper) presets.

The presets are `rcparams.rc_presets` entries, not dicts of their own: a
context is just "these `save.*` rcParams, for this call", which is what
`rc_context` already means.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..rcparams import rc_context, rc_presets, rcParams
from ..utils import file_helpers

contexts = ["poster", "talk", "paper"]


def _save_params() -> dict:
    """The current `save.*` rcParams as `savefig` keywords.

    `None` values are dropped rather than forwarded: `save.pad_inches` is
    unset by default, and passing ``pad_inches=None`` to `savefig` is not the
    same as leaving it out.
    """
    return {key: value for key, value in rcParams.group("save").items() if value is not None}


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
        Named `rc_presets` bundle of `save.*` overrides (dpi, transparency,
        padding) to apply for this call.
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

    # Combine the current save.* rcParams -- through the context preset, if
    # one was named -- with any provided kwargs.
    with rc_context(rc_presets[context] if context is not None else {}):
        save_params = _save_params()
    save_params.update(kwargs)

    # Create folder if it doesn't exist
    folder.mkdir(parents=True, exist_ok=True)
    if move_old:
        file_helpers.move_files_with_glob(folder, f"*{filename}.*")

    if prepend_date:
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{date_str}_{filename}"
    file_extension = save_params.get("format", rcParams["save.format"])
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


__all__ = [
    "contexts",
    "save_figure",
    "save_current_figure",
]
