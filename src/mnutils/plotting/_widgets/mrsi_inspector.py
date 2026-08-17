import base64
from pathlib import Path

from ._html import render_html
from ._shared import load_css, load_esm

_DIR = Path(__file__).parent
_JS = load_esm(_DIR / "mrsi_inspector.js")
_CSS = load_css(_DIR / "mrsi_inspector.css")


class MRSIVoxelInspectorWidget:
    """Interactive T1 + MRSI overlay with click/keyboard voxel selection.

    Every anatomical slice is a pre-rendered WebP frame (T1 + MRSI overlay, no
    voxel box baked in). The voxel outline and the spectrum plot are drawn
    client-side from data embedded directly in the displayed HTML: the
    display affine (for projecting a voxel index to display pixel
    coordinates, and for inverting a click back to a voxel index) and a
    compressed spectra buffer (one spectrum per (x, y, MRSI slice) triple, in
    the units already selected by the caller -- magnitude, autophased, or
    real). None of this requires a Python kernel after the widget is
    displayed.
    """

    def __init__(
        self,
        *,
        left_frames: list[bytes],
        slice_titles: list[str],
        n_anat_slices: int,
        initial_slice: int,
        image_width: int,
        image_height: int,
        mrsi_to_display_affine: list[float],
        display_to_mrsi_affine: list[float],
        grid_shape: tuple[int, int, int],
        mrsi_dims: tuple[int, int],
        initial_voxel: tuple[int, int, int],
        ppm: list[float],
        spectra_bytes: bytes,
        spectra_scale: float,
        npts: int,
        spectrum_label: str = "Spectrum",
    ) -> None:
        self._data = {
            "left_frames": [base64.b64encode(frame).decode("ascii") for frame in left_frames],
            "slice_titles": slice_titles,
            "n_anat_slices": n_anat_slices,
            "initial_slice": initial_slice,
            "image_width": image_width,
            "image_height": image_height,
            "mrsi_to_display_affine": mrsi_to_display_affine,
            "display_to_mrsi_affine": display_to_mrsi_affine,
            "grid_shape": list(grid_shape),
            "mrsi_dims": list(mrsi_dims),
            "initial_voxel": list(initial_voxel),
            "ppm": ppm,
            "spectra_bytes": base64.b64encode(spectra_bytes).decode("ascii"),
            "spectra_scale": spectra_scale,
            "npts": npts,
            "spectrum_label": spectrum_label,
            # Tells the shared frontend how to decode what it just received.
            # The anywidget backend renders the same component with the same
            # keys but sets `transport` to "binary", where the two heavy fields
            # arrive as ArrayBuffers over the comm instead of base64 strings.
            "transport": "base64",
            "frame_mime": "image/webp",
            "spectra_encoding": "zlib-float16",
        }

    def _repr_html_(self) -> str:
        return render_html(_JS, _CSS, self._data)
