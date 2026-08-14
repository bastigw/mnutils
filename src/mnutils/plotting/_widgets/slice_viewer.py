import base64
from pathlib import Path

from ._html import render_html
from ._shared import load_css, load_esm

_DIR = Path(__file__).parent
_JS = load_esm(_DIR / "slice_viewer.js")
_CSS = load_css(_DIR / "slice_viewer.css")


class SliceViewerWidget:
    """Slider-driven viewer over a stack of pre-rendered PNG frames.

    All frames are embedded directly in the displayed HTML, so the slider
    works purely client-side -- no Python kernel is needed after the widget
    is displayed, which is what makes it survive a static docs build.
    """

    def __init__(
        self,
        frames: list[bytes],
        n_slices: int,
        initial_index: int = 0,
        slice_label: str = "Slice",
    ) -> None:
        self._data = {
            "frames": [base64.b64encode(frame).decode("ascii") for frame in frames],
            "n_slices": n_slices,
            "initial_index": initial_index,
            "slice_label": slice_label,
        }

    def _repr_html_(self) -> str:
        return render_html(_JS, _CSS, self._data)
